
import re
from collections import defaultdict
from flair.data import Sentence

from grammarRefiner.grammar_refiner import refine_sentence


""""""

def clean_sentence(sentence, start, end, tagger):
    """"""
    #
    flair_sentence = Sentence(sentence)
    tagger.predict(flair_sentence)


    for entity in flair_sentence.get_spans('ner'):
        if (entity.start_position < end and entity.end_position > start and
                not (entity.start_position >= start and entity.end_position <= end)):
            print(f"PROTECTION TRIGGERED: Entity '{entity.text}' ({entity.start_position},{entity.end_position}) would be affected")
            return sentence

    print("No protection needed, proceeding with deletion...")

    #
    pre_text = sentence[:start].rstrip()
    pre_start = start
    post_text = sentence[end:]

    #
    title_pattern = r'\b(Mr|Mrs|Ms|Dr|Prof|Sir)\s+$'
    title_match = re.search(title_pattern, pre_text)
    if title_match:
        #
        pre_start = title_match.start()
        # print(f"Found title: '{title_match.group(0)}', new pre_start: {pre_start}")

    #
    elif pre_text.endswith(','):
        pre_start = len(pre_text.rstrip(','))
    elif pre_text.endswith(('and', 'or')):
        words = pre_text.split()
        if words:
            last_word = words[-1]
            pre_start = len(pre_text) - len(last_word)

    #
    result = sentence[:pre_start].rstrip() + ' ' + post_text.lstrip()

    #
    result = re.sub(r'\s+', ' ', result)  #
    result = re.sub(r'\s*,\s*', ', ', result)  #

    return result.strip()



""""""


def format_sentence(sentence: str, tagger=None) -> str:
    """Enhanced sentence formatting with improved comma handling for parallel structures."""
    if tagger is None:
        return sentence

    # Step 1: Save special patterns (your existing code)
    special_patterns = {
        'hyphen_words': re.findall(r'\b\w+(?:-\w+)+\b', sentence),
        'numbers': re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', sentence),
        'currencies': re.findall(r'[£€$]\d+(?:,\d{3})*(?:\.\d+)?m?\b', sentence),
        'quotes': re.findall(r'"[^"]+"', sentence),
        'brackets': re.findall(r'\([^)]+\)', sentence),
        'contractions': re.findall(r"\w+n't|\w+'[sd]|\w+'ll|\w+'ve|\w+'re", sentence)
    }

    # Step 2: Basic cleanup first
    result = re.sub(r'\s+', ' ', sentence)  # Normalize spaces

    # Fix punctuation issues
    result = re.sub(r',\s*,', ',', result)  # Remove double commas
    result = re.sub(r',\s+,', ',', result)  # Another pattern for double commas

    # Fix comma before which/that/who
    result = re.sub(r',\s+(which|that|who|where|when)', r' \1', result)

    # Your existing cleanup
    result = re.sub(r'\s*([,.!?])', r'\1', result)  # Clean punctuation spacing
    result = re.sub(r'([,.!?])\s*', r'\1 ', result)  # Add space after punctuation

    # Step 3: Handle comma series with enhanced rules
    # Fix extra commas but preserve "and"/"or"
    result = re.sub(r',+', ',', result)  # Remove duplicate commas
    result = re.sub(r'\s*,\s*', ', ', result)  # Standardize comma spacing

    # Fix cases like "X, and, Y" -> "X and Y" but preserve final "and"
    result = re.sub(r',\s*(?:and|or)\s*,', ' and ', result)

    # Clean up spaces around and/or
    result = re.sub(r'\s+(?:and|or)\s+', ' \\g<0> ', result).strip()

    # Clean up potential artifacts like ", ,"
    result = re.sub(r',\s*,', ',', result)  # Final check for double commas

    # Step 4: Restore special patterns (your existing code)
    for pattern_type, patterns in special_patterns.items():
        for pattern in patterns:
            if pattern_type == 'hyphen_words':
                parts = pattern.split('-')
                wrong_patterns = [
                    r'\b' + r'\s*-\s*'.join(map(re.escape, parts)) + r'\b',
                    r'\b' + r' '.join(map(re.escape, parts)) + r'\b'
                ]
                for wrong_pattern in wrong_patterns:
                    result = re.sub(wrong_pattern, pattern, result)
            elif pattern_type == 'numbers':
                result = re.sub(
                    re.escape(pattern.replace(",", " , ")),
                    pattern,
                    result
                )
            elif pattern_type in ['currencies', 'quotes', 'brackets', 'contractions']:
                if pattern_type == 'contractions':
                    if "n't" in pattern:
                        result = re.sub(r'\b' + re.escape(pattern.replace("n't", " n't")) + r'\b', pattern, result)
                    else:
                        parts = pattern.split("'")
                        result = re.sub(r'\b' + re.escape(parts[0] + " '" + parts[1]) + r'\b', pattern, result)
                else:
                    result = re.sub(re.escape(pattern), pattern, result)

    return result.strip()



def get_entities(sentence_text, tagger):
    """Get entities from sentence using NER tagger."""
    sentence = Sentence(sentence_text)
    tagger.predict(sentence)
    entities = defaultdict(list)

    for entity in sentence.get_spans('ner'):
        entities[entity.tag].append({
            'text': entity.text,
            'start': entity.start_position,
            'end': entity.end_position
        })

    return entities


""""""


def find_coordinated_entities(sentence_text, tagger):
    """"""
    #
    entities = get_entities(sentence_text, tagger)
    coordinated_groups = []

    # Debug:
    print("Identified entities:")
    for entity_type, entity_list in entities.items():
        print(f"{entity_type}:", [f"{e['text']} ({e['start']}, {e['end']})" for e in entity_list])

    #
    for entity_type, entity_list in entities.items():
        if len(entity_list) < 2:
            continue

        #
        ents_sorted = sorted(entity_list, key=lambda x: x['start'])
        i = 0
        while i < len(ents_sorted) - 1:
            current_group = [ents_sorted[i]]
            j = i + 1

            while j < len(ents_sorted):
                #
                text_between = sentence_text[ents_sorted[j - 1]['end']:ents_sorted[j]['start']]

                #
                is_parallel = bool(
                    #
                    re.search(r'^\s*,\s*(and\s+)?$|^\s*,?\s+and\s+$|^\s*,\s*$', text_between, re.IGNORECASE) and
                    #
                    not re.search(r'\b(?:which|that|who|when|where|because)\b', text_between, re.IGNORECASE) and
                    #
                    not re.search(r'\b(?:is|was|are|were|has|have|had)\b', text_between, re.IGNORECASE)
                )

                if is_parallel:
                    current_group.append(ents_sorted[j])
                    j += 1
                else:
                    break

            if len(current_group) > 1:
                # Debug:
                print(f"Found parallel group: {[e['text'] for e in current_group]}")
                coordinated_groups.append({
                    'type': entity_type,
                    'entities': current_group
                })
                i = j
            else:
                i += 1

    return coordinated_groups



def mutate_and_verify(sentence_text, tagger):
    """Generate mutated sentences and verify all entities recognition results."""
    # print("\n=== Debug mutate_and_verify ===")
    # print(f"Original sentence: {sentence_text}")

    coordinated_groups = find_coordinated_entities(sentence_text, tagger)
    mutated_results = []
    suspicious_sentences = []

    # print(f"\nFound {len(coordinated_groups)} coordinated groups:")
    for i, group in enumerate(coordinated_groups):
        # print(f"\nProcessing group {i + 1}:")
        # print(f"Type: {group['type']}")
        # print("Entities:", [f"{e['text']} ({e['start']}, {e['end']})" for e in group['entities']])

        mutated_sentence = sentence_text
        remaining_entity = group['entities'][0]
        # print(f"\nKeeping first entity: {remaining_entity['text']}")
        entity_type = group['type']
        #
        original_entities = get_entities(sentence_text, tagger)

        #
        entities_to_remove = group['entities'][1:]
        # print(f"\nEntities to remove:", [e['text'] for e in entities_to_remove])

        for entity in reversed(entities_to_remove):
            # print(f"\nRemoving entity: {entity['text']}")
            mutated_sentence = clean_sentence(mutated_sentence, entity['start'], entity['end'], tagger)
            # print(f"Sentence after removal: {mutated_sentence}")

        mutated_sentence = refine_sentence(mutated_sentence, tagger)
        mutated_sentence = format_sentence(mutated_sentence)
        #
        result = {
            "mutated_sentence": mutated_sentence,
            "entities": get_entities(mutated_sentence, tagger)
        }
        mutated_results.append(result)

        #
        mutated_entities = get_entities(mutated_sentence, tagger)
        is_suspicious = False
        suspicious_reason = []

        already_checked_entities = set()

        #
        remaining_entity_found = False
        remaining_entity_correct_tag = False

        if entity_type in mutated_entities:
            for mutated_entity in mutated_entities[entity_type]:
                if mutated_entity['text'].strip() == remaining_entity['text'].strip():
                    remaining_entity_found = True
                    remaining_entity_correct_tag = True
                    break

        #
        if not remaining_entity_found:
            for mut_type, mut_entities in mutated_entities.items():
                if mut_type != entity_type:
                    for mutated_entity in mut_entities:
                        if mutated_entity['text'].strip() == remaining_entity['text'].strip():
                            remaining_entity_found = True
                            is_suspicious = True
                            suspicious_reason.append(
                                f"Entity '{remaining_entity['text']}' changed tag from '{entity_type}' to '{mut_type}'"
                            )
                            break
                if remaining_entity_found:
                    break

        #
        if not remaining_entity_found:
            is_suspicious = True
            suspicious_reason.append(
                f"Expected entity '{remaining_entity['text']}' with tag '{entity_type}' not found"
            )

        #
        already_checked_entities.add(remaining_entity['text'])

        #
        for orig_type, orig_entities in original_entities.items():
            for orig_entity in orig_entities:
                #
                if orig_entity['text'] in already_checked_entities:
                    continue

                #
                if any(orig_entity['text'] == e['text'] for e in entities_to_remove):
                    continue

                entity_found = False
                wrong_tag_found = False
                found_tag = None

                for mut_type, mut_entities in mutated_entities.items():
                    for mut_entity in mut_entities:
                        if mut_entity['text'].strip() == orig_entity['text'].strip():
                            entity_found = True
                            if mut_type != orig_type:
                                wrong_tag_found = True
                                found_tag = mut_type
                            break
                    if entity_found:
                        break

                if not entity_found:
                    is_suspicious = True
                    suspicious_reason.append(f"Entity '{orig_entity['text']}' disappeared")
                elif wrong_tag_found:
                    is_suspicious = True
                    suspicious_reason.append(
                        f"Entity '{orig_entity['text']}' changed tag from '{orig_type}' to '{found_tag}'"
                    )

                #
                already_checked_entities.add(orig_entity['text'])

        if is_suspicious:
            suspicious_sentences.append({
                "mutated_sentence": mutated_sentence,
                "original_entities": original_entities,
                "mutated_entities": mutated_entities,
                "reasons": suspicious_reason
            })

    # print("=== End debug ===\n")
    return mutated_results, suspicious_sentences
from flair.data import Sentence
from itertools import product
from collections import defaultdict, Counter
from rules.entity_swap_filters import should_filter_entity_for_swap


def extract_entities(sentence, tagger):
    """
    Extract entities and their tags from a sentence using the NER model.
    """
    sentence_obj = Sentence(sentence)
    tagger.predict(sentence_obj)
    entities = sentence_obj.get_spans("ner")
    entities_with_positions = [
        {"text": entity.text, "tag": entity.tag, "start": entity.start_position, "end": entity.end_position}
        for entity in entities
    ]
    return entities_with_positions

def extract_entity_texts(text, tagger):
    entities = extract_entities(text, tagger)
    return set([entity["text"] for entity in entities])

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if len(union) > 0 else 0





def swap_entities_across_sentences(sentence1, entities1, sentence2, entities2, ner_model="conll3"):

    swapped_sentences = []  
    swapped_entities_list = []  

    filtered_entities1 = []
    filtered_entities2 = []
    for entity in entities1:
        should_filter, reason = should_filter_entity_for_swap(entity, sentence1, ner_model)
        if not should_filter:
            filtered_entities1.append(entity)
    for entity in entities2:
        should_filter, reason = should_filter_entity_for_swap(entity, sentence2, ner_model)
        if not should_filter:
            filtered_entities2.append(entity)

    entities_by_tag = defaultdict(list)
    for e in filtered_entities1:
        entities_by_tag[e["tag"]].append(("s1", e))  # Mark from sentence 1
    for e in filtered_entities2:
        entities_by_tag[e["tag"]].append(("s2", e))  # Mark from sentence 2

    for tag, entities in entities_by_tag.items():
        s1_entities = [e for src, e in entities if src == "s1"]
        s2_entities = [e for src, e in entities if src == "s2"]

        for s1_entity, s2_entity in product(s1_entities, s2_entities):
            swapped_sentence1 = (
                    sentence1[:s1_entity["start"]]
                    + s2_entity["text"]
                    + sentence1[s1_entity["end"]:]
            )
            swapped_sentence2 = (
                    sentence2[:s2_entity["start"]]
                    + s1_entity["text"]
                    + sentence2[s2_entity["end"]:]
            )

            swapped_entities = {
                "entity1_obj": s1_entity,  
                "entity2_obj": s2_entity,  
                "tag": tag,
            }

            swapped_sentences.append((swapped_sentence1, swapped_sentence2))
            swapped_entities_list.append(swapped_entities)

    return swapped_sentences, swapped_entities_list





def _find_entity_in_list(entity_text: str, entity_tag: str, entity_list: list) -> (bool, str):

    found = False
    found_tag = "O"  

    for entity in entity_list:
        if entity["text"] == entity_text:
            found = True
            found_tag = entity["tag"]
            if found_tag == entity_tag:
                return True, entity_tag  # 

    if found:
        return True, found_tag

    return False, "O"


def mutate_and_verify_rule_two(original_sentence1, original_sentence2, tagger, ner_model=None):

    entities1 = extract_entities(original_sentence1, tagger)
    entities2 = extract_entities(original_sentence2, tagger)

    combined_sentence = f"{original_sentence1} {original_sentence2}"

    swapped_sentences, swapped_entities = swap_entities_across_sentences(
        original_sentence1, entities1, original_sentence2, entities2, ner_model
    )

    mutated_results = []
    suspicious_results = []

    original_combined_entities = {}
    for entity in entities1:
        entity_type = entity["tag"]
        if entity_type not in original_combined_entities: original_combined_entities[entity_type] = []
        original_combined_entities[entity_type].append(
            {"text": entity["text"], "start": entity["start"], "end": entity["end"]})
    for entity in entities2:
        entity_type = entity["tag"]
        adjusted_start = len(original_sentence1) + 1 + entity["start"]
        adjusted_end = len(original_sentence1) + 1 + entity["end"]
        if entity_type not in original_combined_entities: original_combined_entities[entity_type] = []
        original_combined_entities[entity_type].append(
            {"text": entity["text"], "start": adjusted_start, "end": adjusted_end})

    for idx, ((swapped_sentence1, swapped_sentence2), swap_info) in enumerate(zip(swapped_sentences, swapped_entities)):
        mutated_combined_sentence = f"{swapped_sentence1} {swapped_sentence2}"

        mutated_entities1 = extract_entities(swapped_sentence1, tagger)
        mutated_entities2 = extract_entities(swapped_sentence2, tagger)

        mutated_combined_entities = {}
        for entity in mutated_entities1:
            entity_type = entity["tag"]
            if entity_type not in mutated_combined_entities: mutated_combined_entities[entity_type] = []
            mutated_combined_entities[entity_type].append(
                {"text": entity["text"], "start": entity["start"], "end": entity["end"]})
        for entity in mutated_entities2:
            entity_type = entity["tag"]
            adjusted_start = len(swapped_sentence1) + 1 + entity["start"]
            adjusted_end = len(swapped_sentence1) + 1 + entity["end"]
            if entity_type not in mutated_combined_entities: mutated_combined_entities[entity_type] = []
            mutated_combined_entities[entity_type].append(
                {"text": entity["text"], "start": adjusted_start, "end": adjusted_end})

        mutated_results.append({
            "original_combined_sentence": combined_sentence,
            "mutated_combined_sentence": mutated_combined_sentence,
            "original_entities": original_combined_entities,
            "mutated_entities": mutated_combined_entities
        })


        reasons = []

        e1_swapped = swap_info["entity1_obj"]
        e2_swapped = swap_info["entity2_obj"]
        e1_swapped_start = e1_swapped["start"]
        e2_swapped_start = e2_swapped["start"]

        for orig_entity in entities1:
            entity_text = orig_entity["text"]
            original_tag = orig_entity["tag"]

            if orig_entity["start"] == e1_swapped_start:
                found, mutated_tag = _find_entity_in_list(entity_text, original_tag, mutated_entities2)

                if not found:
                    reasons.append(f"Entity '{entity_text}' missing from sentence 2 after swapping")
                elif mutated_tag != original_tag:
                    reasons.append(
                        f"Entity '{entity_text}' changed tag from '{original_tag}' to '{mutated_tag}' after swapping to sentence 2")
            else:
                found, mutated_tag = _find_entity_in_list(entity_text, original_tag, mutated_entities1)

                if not found:
                    reasons.append(f"Entity '{entity_text}' missing from sentence 1 (not swapped)")
                elif mutated_tag != original_tag:
                    reasons.append(
                        f"Entity '{entity_text}' changed tag from '{original_tag}' to '{mutated_tag}' in sentence 1 (not swapped)")

        for orig_entity in entities2:
            entity_text = orig_entity["text"]
            original_tag = orig_entity["tag"]

            if orig_entity["start"] == e2_swapped_start:
                found, mutated_tag = _find_entity_in_list(entity_text, original_tag, mutated_entities1)

                if not found:
                    reasons.append(f"Entity '{entity_text}' missing from sentence 1 after swapping")
                elif mutated_tag != original_tag:
                    reasons.append(
                        f"Entity '{entity_text}' changed tag from '{original_tag}' to '{mutated_tag}' after swapping to sentence 1")
            else:
                found, mutated_tag = _find_entity_in_list(entity_text, original_tag, mutated_entities2)

                if not found:
                    reasons.append(f"Entity '{entity_text}' missing from sentence 2 (not swapped)")
                elif mutated_tag != original_tag:
                    reasons.append(
                        f"Entity '{entity_text}' changed tag from '{original_tag}' to '{mutated_tag}' in sentence 2 (not swapped)")

        for mut_entity in mutated_entities1:
            is_expected = False
            if mut_entity["text"] == e2_swapped["text"]:  # 
                is_expected = True
            else:  
                found, _ = _find_entity_in_list(mut_entity["text"], mut_entity["tag"], entities1)
                if found:
                    is_expected = True

            if not is_expected:
                reasons.append(
                    f"Unexpected new entity '{mut_entity['text']}' with tag '{mut_entity['tag']}' appeared in sentence 1")

        for mut_entity in mutated_entities2:
            is_expected = False
            if mut_entity["text"] == e1_swapped["text"]:  # 
                is_expected = True
            else:  # 
                found, _ = _find_entity_in_list(mut_entity["text"], mut_entity["tag"], entities2)
                if found:
                    is_expected = True

            if not is_expected:
                reasons.append(
                    f"Unexpected new entity '{mut_entity['text']}' with tag '{mut_entity['tag']}' appeared in sentence 2")


        if reasons:
            unique_reasons = sorted(list(set(reasons)))

            suspicious_results.append({
                "original_sentence": combined_sentence,
                "mutated_sentence": mutated_combined_sentence,
                "reasons": unique_reasons,  #  'reasons'
                "original_entities": original_combined_entities,
                "mutated_entities": mutated_combined_entities,
                "swapped_entity_type": swap_info["tag"]
            })

    return mutated_results, suspicious_results
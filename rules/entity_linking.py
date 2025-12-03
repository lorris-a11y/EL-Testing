import traceback
from collections import defaultdict

from flair.data import Sentence

from .constants import HONORIFICS
from .spacySimple import comprehensive_checker

# from rules.mutationRule3 import verify_key_entities_consistency, get_entities, \
#     has_multiple_entities_of_same_type

"""
Entity linking and knowledge graph querying with targeted property extraction.
"""

from rules.descriptionProcessor import get_entity_description_conll3
from rules.possessive_utils import needs_possessive_pronoun, detect_gender_in_sentence

import requests
import re
from typing import Tuple, Dict, List
from flair.models import SequenceTagger

import spacy


nlp = spacy.load('en_core_web_sm')

# ------------------------------------------------------
# Entity Recognition Functions
# ------------------------------------------------------

def get_entities(sentence_text, tagger):
    """Use NER model to identify entities in the sentence, returning entity tags and positions."""
    sentence = Sentence(sentence_text)
    tagger.predict(sentence)
    entities = defaultdict(list)

    for entity in sentence.get_spans('ner'):
        entities[entity.tag].append((entity.text, entity.start_position, entity.end_position))

    return entities


def has_multiple_entities_of_same_type(entities):
    """Check if there are multiple entities of the same type."""
    for entity_list in entities.values():
        if len(entity_list) > 1:
            return True
    return False


def verify_key_entities_consistency(original_entities, mutated_entities):
    """
    Verify if key entities from the original sentence are consistent in the mutated sentence.

    Args:
        original_entities (dict): Original entities with types as keys and lists of (text, start, end) as values.
        mutated_entities (dict): Mutated entities with types as keys and lists of (text, start, end) as values.

    Returns:
        dict: Verification results including missing entities, incorrect types, and consistent entities.
    """
    verification_results = {
        "missing_entities": [],  # Entities from the original not found in mutated
        "type_mismatched_entities": [],  # Entities with incorrect types in mutated
        "consistent_entities": []  # Entities correctly recognized in mutated
    }

    for entity_type, entities in original_entities.items():
        mutated_relevant_entities = mutated_entities.get(entity_type, [])
        mutated_texts = {entity[0] for entity in mutated_relevant_entities}  # Ignore positions

        for text, start, end in entities:
            if text not in mutated_texts:
                verification_results["missing_entities"].append((text, entity_type))
            else:
                # Verify type consistency
                for m_text, m_start, m_end in mutated_relevant_entities:
                    if m_text == text and entity_type not in mutated_entities:
                        verification_results["type_mismatched_entities"].append((text, entity_type))

            # Mark as consistent if no issues
            if text in mutated_texts and entity_type in mutated_entities:
                verification_results["consistent_entities"].append((text, entity_type))

    return verification_results



# def setup_knowledge_graph() -> SPARQLWrapper:
#     """Initialize knowledge graph connection."""
#     sparql = SPARQLWrapper("http://dbpedia.org/sparql")
#     sparql.setReturnFormat(JSON)
#     return sparql


def get_fallback_description(entity_type: str) -> str:
    """Get a generic description based on entity type."""
    return {
        "ORG": "an organization",
        "PER": "a person",
        "LOC": "a location",
        "MISC": "an entity"
    }.get(entity_type, "an entity")







def get_entity_description(entity_text: str, entity_type: str, session=None) -> str:
    # return get_improved_entity_description(entity_text, entity_type, session)
    return get_entity_description_conll3(entity_text, entity_type, session)
    # return get_entity_description_dbpedia_conll3(entity_text, entity_type, session)


def extract_description_after_predicate(description: str) -> str:
    """

    "Google LLC is an American company" -> "an American company"
    "Microsoft Corporation was founded" -> ""
    """
    #
    predicates = [
        r'\bis\b', r'\bwas\b', r'\bhas been\b', r'\bhave been\b',
        r'\bbecame\b', r'\bremains\b'
    ]

    #
    pattern = '|'.join(predicates)

    #
    match = re.search(pattern, description, re.IGNORECASE)
    if match:
        #
        predicate_end = match.end()
        extracted = description[predicate_end:].strip()
        #
        if len(extracted) < 3:  #
            return ""
        return extracted

    return ""  #

#new clean_entity_description

def clean_entity_description(description: str) -> str:
    """

    """
    if not description:
        return ""


    description_stripped = description.strip()
    if re.match(r'^(a|an)\s+\w+', description_stripped, re.IGNORECASE):
        return description_stripped

    extracted = extract_description_after_predicate(description)
    if not extracted:
        return ""

    #
    cleaned = extracted.strip(' .,')
    #
    if len(cleaned) < 3:
        return ""

    return cleaned



def combine_entity_and_description(entity_name: str, description: str) -> str:
    """

    """
    if not description:
        return entity_name

    #
    combined = f"{entity_name} is {description}"

    #
    combined = re.sub(r'\s+', ' ', combined)  #
    combined = re.sub(r'\.+', '.', combined)  #
    if not combined.endswith('.'):
        combined += '.'

    return combined


def process_description(raw_description: str, entity_name: str) -> str:
    """

    """
    #
    cleaned_desc = clean_entity_description(raw_description)

    #
    if not cleaned_desc:
        return ""

    #
    return combine_entity_and_description(entity_name, cleaned_desc)






def select_pronoun(entity_type: str, context_after: str = "", sentence: str = "") -> str:
    """

    """
    print(f"\nDEBUG - Selecting pronoun:")
    print(f"Entity type: {entity_type}")
    print(f"Context after: '{context_after}'")

    #
    base_pronouns = {
        "PER": "he",  #
        "ORG": "it",
        "LOC": "it",
        "MISC": "it"
    }

    #
    needs_possessive = needs_possessive_pronoun(context_after, entity_type)
    print(f"DEBUG - Needs possessive: {needs_possessive}")

    if needs_possessive:
        return "its"

    #
    if entity_type == "PER" and sentence:
        detected_gender = detect_gender_in_sentence(sentence)
        print(f"DEBUG - Detected gender: {detected_gender}")

        if detected_gender == 'female':
            return "she"
        else:
            return "he"  #

    return base_pronouns.get(entity_type, "it")


def adjust_sentence_structure(sentence: str) -> str:
    """
    """

    sentence = re.sub(r'\s+', ' ', sentence)

    #
    fixes = [
        #
        (r'Mr\.?\s+he', 'he'),
        (r'Mrs\.?\s+she', 'she'),
        (r'Ms\.?\s+she', 'she'),
        (r'Dr\.?\s+he', 'he'),
        (r'Prof\.?\s+he', 'he'),

        #
        (r'Mr\.?\s+his', 'his'),
        (r'Mrs\.?\s+her', 'her'),
        (r'Ms\.?\s+her', 'her'),
        (r'Dr\.?\s+his', 'his'),
        (r'Prof\.?\s+his', 'his'),

        #
        (r'[Tt]he\s+it\b(?!\s+is)', ' it'),
        (r'[Tt]he\s+its\b', ' its'),
        (r'[Tt]he\s+he\b', ' he'),
        (r'[Tt]he\s+his\b', ' his'),
        (r'[Tt]he\s+she\b', ' she'),
        (r'[Tt]he\s+her\b', ' her'),

        #
        (r'(?:^|\s)(?:a|an)\s+it\b(?!\s+is)', ' it'),
        (r'(?:^|\s)(?:a|an)\s+its\b', ' its'),
        (r'(?:^|\s)(?:a|an)\s+he\b', ' he'),
        (r'(?:^|\s)(?:a|an)\s+his\b', ' his'),
        (r'(?:^|\s)(?:a|an)\s+she\b', ' she'),
        (r'(?:^|\s)(?:a|an)\s+her\b', ' her'),
    ]

    result = sentence
    for pattern, replacement in fixes:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)




    prepositions = ['at', 'in', 'on', 'by', 'with', 'from', 'to', 'for', 'of', 'through']
    for prep in prepositions:
        result = re.sub(fr'\b{prep}\b\s*([a-zA-Z])', fr'{prep} \1', result, flags=re.IGNORECASE)


    #
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s+([,.!?])', r'\1', result)

    return result.strip()



class EntitySpan:
    """Class to hold entity span information"""

    def __init__(self, start: int, end: int, honorific: str = None, article: str = None):
        self.start = start
        self.end = end
        self.honorific = honorific
        self.article = article

def find_full_entity_span(sentence: str, entity: tuple) -> EntitySpan:
    """


    Args:
        sentence:
        entity: (entity_text, start_pos, end_pos)

    Returns:
        EntitySpan object containing start, end, honorific, and article information
    """
    entity_text, start_pos, end_pos = entity
    prefix_text = sentence[:start_pos].rstrip()

    #
    span = EntitySpan(start_pos, end_pos)

    #
    if prefix_text.endswith("the ") or prefix_text.endswith("The "):
        span.article = "the"
        span.start = start_pos - 4  # "the "

    #
    honorifics = ["Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.", "Dr", "Dr.", "Prof", "Prof."]
    for honorific in honorifics:
        if prefix_text.endswith(honorific + " "):
            span.honorific = honorific
            span.start = start_pos - len(honorific) - 1
            break

    return span

def replace_with_pronoun(sentence: str, entity: tuple, entity_type: str) -> Tuple[str, str]:
    """"""
    try:
        entity_text, _, _ = entity
        span = find_full_entity_span(sentence, entity)

        print("\nDEBUG - Initial values:")
        print(f"Original sentence: {sentence}")
        print(f"Entity: {entity_text}")
        print(f"Span start: {span.start}, end: {span.end}")

        #
        context_after = sentence[span.end:].lstrip()
        print(f"\nDEBUG - Context after entity: '{context_after}'")


        pronoun = select_pronoun(entity_type, context_after, sentence)
        print(f"DEBUG - Selected pronoun: '{pronoun}'")

        #
        full_entity_text = sentence[span.start:span.end].strip()
        print(f"DEBUG - Full entity text: '{full_entity_text}'")

        #
        prefix = sentence[:span.start].rstrip()
        suffix = sentence[span.end:].lstrip()
        print(f"\nDEBUG - Before space processing:")
        print(f"Prefix: '{prefix}'")
        print(f"Suffix: '{suffix}'")

        #
        if prefix:
            words = prefix.split()
            if words:
                last_word = words[-1].lower()
                prepositions = {'at', 'in', 'on', 'by', 'with', 'from', 'to', 'for', 'of', 'through'}

                print(f"DEBUG - Last word of prefix: '{last_word}'")
                print(f"DEBUG - Is preposition: {last_word in prepositions}")

                if last_word in prepositions:
                    #
                    prefix = ' '.join(words[:-1] + [last_word]) + ' '
                    print(f"DEBUG - Modified prefix after preposition: '{prefix}'")
                elif not prefix[-1].isspace():
                    prefix += ' '
                    print(f"DEBUG - Added space to prefix: '{prefix}'")

        #
        if suffix and not suffix[0] in " \n\t.,!?":
            suffix = ' ' + suffix
            print(f"DEBUG - Modified suffix with space: '{suffix}'")




        print(f"Pronoun before capitalization: '{pronoun}'")
        #
        needs_capitalization = not prefix or prefix.rstrip()[-1] in ".!?\n"
        print(f"Needs capitalization: {needs_capitalization}")
        if needs_capitalization:
            pronoun = pronoun.capitalize()  # 这应该对任何代词都生效，包括"its"
            print(f"Pronoun after capitalization: '{pronoun}'")

        #
        new_sentence = prefix + pronoun + suffix
        print(f"\nDEBUG - Before final cleanup:")
        print(f"New sentence: '{new_sentence}'")

        #
        new_sentence = adjust_sentence_structure(new_sentence)

        print(f"\nDEBUG - Final result:")
        print(f"Final sentence: '{new_sentence}'")

        return new_sentence, full_entity_text

    except Exception as e:
        print(f"Error in replace_with_pronoun: {str(e)}")
        traceback.print_exc()
        return sentence, entity_text





def should_skip_entity_replacement(sentence: str, entity: tuple, entity_type: str = None) -> bool:
    """

    """
    entity_text, start_pos, end_pos = entity

    print(f"DEBUG - ConLL3 hybrid check for '{entity_text}' ({entity_type})")

    # === 1.  ===
    if comprehensive_checker.should_skip_entity_replacement(sentence, entity, entity_type, "conll3"):
        print(f"DEBUG - ConLL3: Skipped by comprehensive_checker")
        return True

    # === 2.  ===
    text_before = sentence[:start_pos].rstrip()

    if text_before:
        #
        words_before = text_before.split()
        if words_before:
            last_word = words_before[-1].rstrip('.,!?;:')
            if last_word in HONORIFICS:  #
                print(f"DEBUG - ConLL3 specific: Found honorific '{last_word}' before entity '{entity_text}'")
                return True


    print(f"DEBUG - ConLL3: All checks passed, allowing replacement for '{entity_text}'")
    return False


def mutate_and_verify_with_knowledge_graph(sentence_text: str, tagger: SequenceTagger, dbpedia_session=None) -> Tuple[
    List[Dict], List[Dict]]:
    """"""
    try:
        entities = get_entities(sentence_text, tagger)


        # 检查是否有任何实体
        has_entities = any(len(entity_list) > 0 for entity_list in entities.values())
        if not has_entities:
            return [], []

        #
        if dbpedia_session is None:
            dbpedia_session = requests.Session()
            dbpedia_session.proxies = {
                "http": "Your proxy here",
                "https": "Your proxy here",
            }
            dbpedia_session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            })

        mutated_results = []
        suspicious_sentences = []



        for entity_type, entity_list in entities.items():
            if len(entity_list) == 0:  #
                continue

            try:
                first_entity = entity_list[0]

                #
                if should_skip_entity_replacement(sentence_text, first_entity):
                    continue

                mutated_sentence, entity_name = replace_with_pronoun(
                    sentence_text, first_entity, entity_type
                )

                #
                raw_description = get_entity_description(
                    entity_text=entity_name,
                    entity_type=entity_type,
                    session=dbpedia_session
                )

                entity_intro = process_description(raw_description, entity_name)

                if not entity_intro:
                    entity_intro = f"{entity_name} is {get_fallback_description(entity_type)}."

                combined_sentence = f"{entity_intro} {mutated_sentence}"
                combined_entities = get_entities(combined_sentence, tagger)

                result = {
                    "mutated_sentence": combined_sentence,
                    "entities": combined_entities,
                    "original_text": sentence_text
                }
                mutated_results.append(result)

                #
                verification = verify_key_entities_consistency(
                    original_entities=entities,
                    mutated_entities=combined_entities
                )

                if verification["missing_entities"] or verification["type_mismatched_entities"]:
                    #
                    reasons = []

                    #
                    for entity_text, entity_type in verification["missing_entities"]:
                        reasons.append(f"Entity '{entity_text}' of type '{entity_type}' is missing")

                    #
                    for entity_text, entity_type in verification["type_mismatched_entities"]:
                        #
                        mutated_type = None
                        for m_type, m_entities in combined_entities.items():
                            for m_text, _, _ in m_entities:
                                if m_text == entity_text:
                                    mutated_type = m_type
                                    break
                            if mutated_type:
                                break

                        if mutated_type:
                            reasons.append(
                                f"Entity '{entity_text}' changed tag from '{entity_type}' to '{mutated_type}'")
                        else:
                            reasons.append(f"Entity '{entity_text}' changed tag from '{entity_type}'")

                    #
                    suspicious_sentences.append({
                        "original_sentence": sentence_text,
                        "mutated_sentence": combined_sentence,
                        "reasons": reasons,
                        "original_entities": entities,
                        "mutated_entities": combined_entities
                    })

            except Exception as e:
                print(f"Error processing entity {first_entity}: {str(e)}")
                traceback.print_exc()
                continue

        return mutated_results, suspicious_sentences

    except Exception as e:
        print(f"Error in mutation process: {str(e)}")
        traceback.print_exc()
        return [], []





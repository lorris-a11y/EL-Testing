import re
import traceback
from typing import Dict, Tuple, List

import requests
#
import spacy
from flair.models import SequenceTagger

from rules.descriptionProcessor import MultiNEREntityProcessor, get_entity_description_ontonotes
#
from rules.entity_linking import (
    process_description,
    # should_skip_entity_replacement,
    find_full_entity_span,
    adjust_sentence_structure,
    get_entities, verify_key_entities_consistency
)
from rules.possessive_utils import needs_possessive_pronoun, detect_gender_in_sentence
from rules.spacySimple import should_skip_entity_replacement_ontonotes

#
# from rules.mutationRule3 import (
#     verify_key_entities_consistency,
#     get_entities,
#     has_multiple_entities_of_same_type
# )

try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Warning: spaCy model 'en_core_web_sm' not loaded. Using simplified possessive checks.")
    nlp = None


def get_fallback_description(entity_type: str) -> str:
    """Get a generic description based on entity type for Flair-Ontonotes."""
    return {
        # Person types
        "PERSON": "a person",

        # Organization types
        "ORG": "an organization",
        "GPE": "a geopolitical entity",

        # Location types
        "LOC": "a location",
        "FAC": "a facility",

        # Event types
        "EVENT": "an event",

        # Product types
        "PRODUCT": "a product",
        "WORK_OF_ART": "a work of art",

        # Document types
        "LAW": "a law",

        # Numeric types
        "DATE": "a date",
        "TIME": "a time",
        "PERCENT": "a percentage",
        "MONEY": "a monetary amount",
        "QUANTITY": "a quantity",
        "ORDINAL": "an ordinal number",
        "CARDINAL": "a cardinal number",

        # Other types
        "NORP": "a nationality, religious, or political group",
        "LANGUAGE": "a language",

        # Catch-all
        "MISC": "an entity"
    }.get(entity_type, "an entity")


def should_skip_entity_replacement(sentence: str, entity: tuple, entity_type: str) -> bool:
    #
    return should_skip_entity_replacement_ontonotes(sentence, entity, entity_type)

def _simple_possessive_check(text_after_entity: str) -> bool:
    """

    """
    text = text_after_entity.lstrip()


    if re.match(r'^(?:is|was|were|are|be)\b', text, re.IGNORECASE):
        return False

    prepositions = r'^(?:in|on|at|by|with|from|to|for|of|through)\s+'
    if re.match(prepositions, text, re.IGNORECASE):
        return False

    determiners = r'^(?:the|a|an)\s+'
    if re.match(determiners, text, re.IGNORECASE):
        return False

    conjunctions = r'^(?:and|or|but|if|because|when|while)\s+'
    if re.match(conjunctions, text, re.IGNORECASE):
        return False

    relatives = r'^(?:which|that|who|whom|whose)\s+'
    if re.match(relatives, text, re.IGNORECASE):
        return False

    return bool(re.match(r'^[a-zA-Z]+\b', text))





def select_pronoun(entity_type: str, context_after: str = "", sentence: str = "") -> str:

    print(f"\nDEBUG - Selecting pronoun:")
    print(f"Entity type: {entity_type}")
    print(f"Context after: '{context_after}'")

    #
    base_pronouns = {
        "PERSON": "he",

        "ORG": "it",
        "GPE": "it",

        # Location types
        "LOC": "it",
        "FAC": "it",

        "EVENT": "it",

        "PRODUCT": "it",
        "WORK_OF_ART": "it",

        "LAW": "it",

        "NORP": "they",  #

        #
        "MISC": "it"
    }

    #
    needs_possessive = needs_possessive_pronoun(context_after, entity_type)
    print(f"DEBUG - Needs possessive: {needs_possessive}")

    if needs_possessive:
        #
        possessive_pronouns = {
            "PERSON": "their",  #
            "NORP": "their",
        }
        return possessive_pronouns.get(entity_type, "its")

    #
    if entity_type == "PERSON" and sentence:
        detected_gender = detect_gender_in_sentence(sentence)
        print(f"DEBUG - Detected gender: {detected_gender}")

        if detected_gender == 'female':
            return "she"
        else:
            return "he"  #

    return base_pronouns.get(entity_type, "it")


def replace_with_pronoun(sentence: str, entity: tuple, entity_type: str) -> Tuple[str, str]:
    """"""
    try:
        entity_text, _, _ = entity
        span = find_full_entity_span(sentence, entity)

        print("\nDEBUG - Initial values:")
        print(f"Original sentence: {sentence}")
        print(f"Entity: {entity_text}")
        print(f"Span start: {span.start}, end: {span.end}")

        context_after = sentence[span.end:].lstrip()
        print(f"\nDEBUG - Context after entity: '{context_after}'")

        # pronoun = select_pronoun(entity_type, context_after)

        pronoun = select_pronoun(entity_type, context_after, sentence)
        print(f"DEBUG - Selected pronoun: '{pronoun}'")

        full_entity_text = sentence[span.start:span.end].strip()

        prefix = sentence[:span.start].rstrip()
        suffix = context_after

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

        #
        needs_capitalization = not prefix or prefix.rstrip()[-1] in ".!?\n"
        print(f"Needs capitalization: {needs_capitalization}")
        if needs_capitalization:
            pronoun = pronoun.capitalize()
            print(f"Pronoun after capitalization: '{pronoun}'")

        #
        new_sentence = prefix + pronoun + suffix
        print(f"\nDEBUG - Before final cleanup:")
        print(f"New sentence: '{new_sentence}'")

        return adjust_sentence_structure(new_sentence), full_entity_text

    except Exception as e:
        print(f"Error in replace_with_pronoun: {str(e)}")
        traceback.print_exc()
        return sentence, entity_text




def get_ontonotes_fallback_description(entity_type: str) -> str:
    """Get fallback description for OntoNotes entity types."""
    processor = MultiNEREntityProcessor()
    return processor._get_fallback_description(entity_type, "ontonotes")

def get_entity_description(entity_text: str, entity_type: str, session=None) -> str:
    """Get entity description from knowledge graph or fallback to generic description."""
    #
    if entity_type in {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}:
        return get_ontonotes_fallback_description(entity_type)

    return get_entity_description_ontonotes(entity_text, entity_type, session)



def mutate_and_verify_with_knowledge_graph(sentence_text: str, tagger: SequenceTagger, dbpedia_session=None) -> Tuple[
    List[Dict], List[Dict]]:
    """"""
    try:
        entities = get_entities(sentence_text, tagger)

        #
        has_entities = any(len(entity_list) > 0 for entity_list in entities.values())
        if not has_entities:
            return [], []

        #
        if dbpedia_session is None:
            dbpedia_session = requests.Session()
            dbpedia_session.proxies = {
                "http": "http://127.0.0.1:7890",
                "https": "http://127.0.0.1:7890",
            }
            dbpedia_session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            })

        mutated_results = []
        suspicious_sentences = []

        #
        for entity_type, entity_list in entities.items():
            if len(entity_list) == 0:  #
                continue

            try:
                first_entity = entity_list[0]

                #
                if should_skip_entity_replacement(sentence_text, first_entity, entity_type):
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

                    reasons = []


                    for entity_text, entity_type in verification["missing_entities"]:
                        reasons.append(f"Entity '{entity_text}' of type '{entity_type}' is missing")


                    for entity_text, entity_type in verification["type_mismatched_entities"]:

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
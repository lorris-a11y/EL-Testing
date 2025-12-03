
import re
import traceback
from collections import defaultdict
from typing import Dict, Tuple, List

import requests
import spacy
from flair.data import Sentence

from rules.descriptionProcessor import get_entity_description_aws
# Import the necessary functions from your original implementation
from rules.entity_linking import (
    process_description,
    adjust_sentence_structure,
    find_full_entity_span, verify_key_entities_consistency
)
# Import verification functions
# from rules.mutationRule3 import (
#     verify_key_entities_consistency
# )
from rules.possessive_utils import needs_possessive_pronoun, detect_gender_in_sentence
from rules.spacySimple import should_skip_entity_replacement_aws

# Load spaCy model
nlp = spacy.load('en_core_web_sm')



def get_entities(sentence_text, tagger):
    """Use AWS NER model to identify entities in the sentence, returning entity tags and positions."""
    sentence = Sentence(sentence_text)
    tagger.predict(sentence)
    entities = defaultdict(list)

    for entity in sentence.get_spans('ner'):
        entities[entity.tag].append((entity.text, entity.start_position, entity.end_position))

    return entities



def get_aws_fallback_description(entity_type: str) -> str:
    return {
        "PERSON": "a person",
        "TITLE": "a creative work",  #
        "ORGANIZATION": "an organization",
        "COMMERCIAL_ITEM": "a product",  #
        "LOCATION": "a location",
        "EVENT": "an event",
        "DATE": "a date",
        "QUANTITY": "a quantity",
        "OTHER": "an entity"
    }.get(entity_type, "an entity")


def _simple_possessive_check(text_after_entity: str) -> bool:
    """
    """
    text = text_after_entity.lstrip()

    #
    if re.match(r'^(?:is|was|were|are|be)\b', text, re.IGNORECASE):
        return False

    #
    prepositions = r'^(?:in|on|at|by|with|from|to|for|of|through)\s+'
    if re.match(prepositions, text, re.IGNORECASE):
        return False

    #
    determiners = r'^(?:the|a|an)\s+'
    if re.match(determiners, text, re.IGNORECASE):
        return False

    #
    conjunctions = r'^(?:and|or|but|if|because|when|while)\s+'
    if re.match(conjunctions, text, re.IGNORECASE):
        return False

    #
    relatives = r'^(?:which|that|who|whom|whose)\s+'
    if re.match(relatives, text, re.IGNORECASE):
        return False

    #
    return bool(re.match(r'^[a-zA-Z]+\b', text))



def select_pronoun(entity_type: str, context_after: str = "", sentence: str = "") -> str:
    """Select appropriate pronoun based on AWS entity type and context."""
    print(f"\nDEBUG - Selecting pronoun:")
    print(f"Entity type: {entity_type}")
    print(f"Context after: '{context_after}'")

    base_pronouns = {
        "PERSON": "he",  #
        "TITLE": "it",
        "ORGANIZATION": "it",
        "COMMERCIAL_ITEM": "it",
        "LOCATION": "it",
        "EVENT": "it",
        "DATE": "it",
        "QUANTITY": "it",
        "OTHER": "it"
    }

    # Check if possessive form is needed
    needs_possessive = needs_possessive_pronoun(context_after, entity_type)
    print(f"DEBUG - Needs possessive: {needs_possessive}")

    if needs_possessive:
        # Define possessive pronouns
        possessive_pronouns = {
            "PERSON": "their",  #
            "TITLE": "its",
        }
        return possessive_pronouns.get(entity_type, "its")

    if entity_type == "PERSON" and sentence:
        detected_gender = detect_gender_in_sentence(sentence)
        print(f"DEBUG - Detected gender: {detected_gender}")

        if detected_gender == 'female':
            return "she"
        else:
            return "he"

    return base_pronouns.get(entity_type, "it")



def should_skip_entity_replacement(sentence: str, entity: tuple, entity_type: str) -> bool:
    # AWS
    return should_skip_entity_replacement_aws(sentence, entity, entity_type)


def replace_with_pronoun(sentence: str, entity: tuple, entity_type: str) -> Tuple[str, str]:
    """
    Replace entity with appropriate pronoun based on type and context.
    """
    try:
        entity_text, _, _ = entity
        span = find_full_entity_span(sentence, entity)

        print("\nDEBUG - Initial values:")
        print(f"Original sentence: {sentence}")
        print(f"Entity: {entity_text}")
        print(f"Span start: {span.start}, end: {span.end}")

        # Get context after entity
        context_after = sentence[span.end:].lstrip()
        print(f"\nDEBUG - Context after entity: '{context_after}'")

        # Select appropriate pronoun
        # pronoun = select_pronoun(entity_type, context_after)

        pronoun = select_pronoun(entity_type, context_after, sentence)
        print(f"DEBUG - Selected pronoun: '{pronoun}'")

        # Get full entity text including any honorifics or articles
        full_entity_text = sentence[span.start:span.end].strip()
        print(f"DEBUG - Full entity text: '{full_entity_text}'")

        # Handle spacing and punctuation
        prefix = sentence[:span.start].rstrip()
        suffix = sentence[span.end:].lstrip()
        print(f"\nDEBUG - Before space processing:")
        print(f"Prefix: '{prefix}'")
        print(f"Suffix: '{suffix}'")

        # Special handling for prepositions
        if prefix:
            words = prefix.split()
            if words:
                last_word = words[-1].lower()
                prepositions = {'at', 'in', 'on', 'by', 'with', 'from', 'to', 'for', 'of', 'through'}

                print(f"DEBUG - Last word of prefix: '{last_word}'")
                print(f"DEBUG - Is preposition: {last_word in prepositions}")

                if last_word in prepositions:
                    # Ensure space after preposition
                    prefix = ' '.join(words[:-1] + [last_word]) + ' '
                    print(f"DEBUG - Modified prefix after preposition: '{prefix}'")
                elif not prefix[-1].isspace():
                    prefix += ' '
                    print(f"DEBUG - Added space to prefix: '{prefix}'")

        # Handle spacing in suffix
        if suffix and not suffix[0] in " \n\t.,!?":
            suffix = ' ' + suffix
            print(f"DEBUG - Modified suffix with space: '{suffix}'")

        # Handle capitalization
        print(f"Pronoun before capitalization: '{pronoun}'")
        needs_capitalization = not prefix or prefix.rstrip()[-1] in ".!?\n"
        print(f"Needs capitalization: {needs_capitalization}")
        if needs_capitalization:
            pronoun = pronoun.capitalize()
            print(f"Pronoun after capitalization: '{pronoun}'")

        # Build new sentence and adjust structure
        new_sentence = prefix + pronoun + suffix
        print(f"\nDEBUG - Before final cleanup:")
        print(f"New sentence: '{new_sentence}'")

        new_sentence = adjust_sentence_structure(new_sentence)
        print(f"\nDEBUG - Final result:")
        print(f"Final sentence: '{new_sentence}'")

        return new_sentence, full_entity_text

    except Exception as e:
        print(f"Error in replace_with_pronoun: {str(e)}")
        traceback.print_exc()
        return sentence, entity_text



def get_entity_description(entity_text: str, entity_type: str, session=None) -> str:
    """Get entity description from knowledge graph or fallback to generic description."""
    if entity_type in {"DATE", "QUANTITY"}:
        return get_aws_fallback_description(entity_type)

    return get_entity_description_aws(entity_text, entity_type, session)

def mutate_and_verify_with_knowledge_graph(sentence_text: str, tagger, dbpedia_session=None) -> Tuple[
    List[Dict], List[Dict]]:
    """
    Mutate sentence by replacing entities with pronouns and adding descriptions.

    Args:
        sentence_text (str): The original sentence
        tagger: The NER tagger object
        dbpedia_session (requests.Session, optional): Session for DBpedia requests
    """
    try:
        # Get entities using the tagger
        entities = get_entities(sentence_text, tagger)

        has_entities = any(len(entity_list) > 0 for entity_list in entities.values())
        if not has_entities:
            return [], []

        # Create a session specifically for DBpedia if none provided
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

        for entity_type, entity_list in entities.items():

            if len(entity_list) == 0:  #
                continue

            try:
                first_entity = entity_list[0]

                # Skip if needed - pass entity_type to the function
                if should_skip_entity_replacement(sentence_text, first_entity, entity_type):
                    continue

                # Replace with pronoun
                mutated_sentence, entity_name = replace_with_pronoun(
                    sentence_text, first_entity, entity_type
                )

                # Get entity description using the DBpedia session
                raw_description = get_entity_description(
                    entity_name, entity_type, session=dbpedia_session
                )

                # Process the description
                entity_intro = process_description(raw_description, entity_name)

                if not entity_intro:
                    entity_intro = f"{entity_name} is {get_aws_fallback_description(entity_type)}."

                # Create the combined sentence
                combined_sentence = f"{entity_intro} {mutated_sentence}"
                combined_entities = get_entities(combined_sentence, tagger)

                result = {
                    "mutated_sentence": combined_sentence,
                    "entities": combined_entities,
                    "original_text": sentence_text
                }
                mutated_results.append(result)

                # Verify entity consistency
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
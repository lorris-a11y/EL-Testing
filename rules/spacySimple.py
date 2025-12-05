import spacy
from typing import Tuple

from rules.entity_skip_checks import (
    has_spacy_noun_before_ner_entity,
    is_entity_in_quotes,
    is_entity_in_formatted_brackets, has_spacy_noun_adjacent_to_ner_entity, has_pronoun_before_entity_spacy,
    has_definite_article_before_entity, has_possessive_before_entity, has_number_before_entity,
    has_ordinal_before_entity, has_wh_problem_before_entity_spacy
)

from .constants import GENERIC_ENTITIES, POSITION_TITLES

class ComprehensiveEntityChecker:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
            print("DEBUG - spaCy model loaded successfully for comprehensive checking")
        except:
            self.nlp = None
            print("Warning: spaCy model not available")

    def should_skip_entity_replacement(self, sentence: str, entity: Tuple[str, int, int],
                                       entity_type: str = None, ner_model: str = "aws") -> bool:

        entity_text, start_pos, end_pos = entity

        print(f"DEBUG - Comprehensive check for '{entity_text}' ({entity_type}) [Model: {ner_model}]")

        if self._check_abbreviation_in_parentheses(sentence, entity):
            return True

        if self._check_non_replaceable_types(entity_type, ner_model):
            return True

        if self._check_generic_entities(entity_text):
            return True

        if is_entity_in_quotes(sentence, entity):
            print(f"DEBUG - Import check: Entity in quotes, skipping")
            return True

        if is_entity_in_formatted_brackets(sentence, entity):
            print(f"DEBUG - Import check: Entity in formatted brackets, skipping")
            return True

        if has_pronoun_before_entity_spacy(sentence, entity):
            print(f"DEBUG - Import check: Pronoun before entity, skipping")
            return True

        if has_spacy_noun_adjacent_to_ner_entity(sentence, entity):
            print(f"DEBUG - Import check: Noun adjacent to entity, skipping")
            return True

        if has_number_before_entity(sentence, entity):
            print(f"DEBUG - Import check: Number before entity, skipping")
            return True

        if has_ordinal_before_entity(sentence, entity):
            print(f"DEBUG - Import check: Ordinal before entity, skipping")
            return True

        if has_wh_problem_before_entity_spacy(sentence, entity):
            print(f"DEBUG - spaCy check: Problematic pronoun before entity, skipping")
            return True


        if self._check_dependency_syntax(sentence, entity, entity_type):
            return True

        if self._check_problematic_preposition_pattern(sentence, entity, entity_type):
            return True

        if self._check_title_patterns_simple(sentence, entity):
            return True

        if self._check_entity_overlap_simple(sentence, entity):
            return True

        if self._check_capitalized_word_after_entity(sentence, entity):
            print(f"DEBUG - Entity followed by capitalized word, skipping")
            return True

        if has_definite_article_before_entity(sentence, entity):
            print(f"DEBUG - Entity preceded by 'the', skipping to avoid 'the it' pattern")
            return True

        if has_possessive_before_entity(sentence, entity):
            print(f"DEBUG - Entity preceded by possessive, skipping to avoid \"'s it\" pattern")
            return True

        if self._check_naming_verb_pattern(sentence, entity):
            return True

        if self._check_honorific_before_entity(sentence, entity):
            return True

        print(f"DEBUG - All checks passed, allowing replacement for '{entity_text}'")
        return False



    def _check_abbreviation_in_parentheses(self, sentence: str, entity: Tuple[str, int, int]) -> bool:
        entity_text, start_pos, end_pos = entity

        text_after = sentence[end_pos:].lstrip()
        if text_after.startswith('('):
            bracket_end = text_after.find(')')
            if bracket_end != -1:
                abbrev = text_after[1:bracket_end].strip()
                is_abbrev = abbrev.isupper() or '.' in abbrev or len(abbrev) <= 5
                if is_abbrev:
                    print(f"DEBUG - Original check: Skipping entity with abbreviation: {entity_text} ({abbrev})")
                    return True
        return False

    def _check_entity_overlap_simple(self, sentence: str, entity: Tuple[str, int, int]) -> bool:
        if self.nlp is None:
            print("DEBUG - spaCy not available for entity overlap check")
            return False

        entity_text, start_pos, end_pos = entity

        try:
            doc = self.nlp(sentence)

            print(f"DEBUG - Checking overlap for '{entity_text}' at {start_pos}-{end_pos}")
            print(
                f"DEBUG - spaCy entities: {[(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}")

            for ent in doc.ents:
                ent_start = ent.start_char
                ent_end = ent.end_char

                has_overlap = (ent_start <= start_pos < ent_end or
                               start_pos <= ent_start < end_pos)
                is_same = (ent_start == start_pos and ent_end == end_pos)

                if has_overlap and not is_same:
                    print(
                        f"DEBUG - Entity overlap: '{entity_text}' is part of larger entity '{ent.text}' ({ent.label_})")
                    return True

            print(f"DEBUG - No overlap found for '{entity_text}'")
            return False

        except Exception as e:
            print(f"DEBUG - Entity overlap check failed: {e}")
            return False


    def _check_non_replaceable_types(self, entity_type: str, ner_model: str) -> bool:
        if ner_model.lower() == "aws":
            non_replaceable_types = {"DATE", "QUANTITY"}
        elif ner_model.lower() == "azure":
            non_replaceable_types = {
                "DateTime",  # 
                "PhoneNumber",  # 
                "Email",  # 
                "URL",  # 
                "IP",  # 
                "Quantity",  # 
                "Address",  # 
                "Skill",   # 
                "PersonType",  # 
            }
        elif ner_model.lower() == "ontonotes":
            non_replaceable_types = {
                "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY",
                "ORDINAL", "CARDINAL", "LANGUAGE"
            }
        elif ner_model.lower() == "conll3":
            non_replaceable_types = set()
        else:
            # 
            non_replaceable_types = {"DATE", "QUANTITY"}

        if entity_type in non_replaceable_types:
            print(f"DEBUG - {ner_model.upper()} check: Skipping non-replaceable entity type: {entity_type}")
            return True
        return False

    def _check_honorific_before_entity(self, sentence: str, entity: Tuple[str, int, int]) -> bool:

        from rules.constants import HONORIFICS

        entity_text, start_pos, end_pos = entity
        text_before = sentence[:start_pos].rstrip()

        if text_before:
            words_before = text_before.split()
            if words_before:
                last_word = words_before[-1].rstrip('.,!?;:')
                if last_word in HONORIFICS:
                    print(f"DEBUG - Found honorific '{last_word}' before entity '{entity_text}', skipping")
                    return True

        return False


    def _check_generic_entities(self, entity_text: str) -> bool:
        entity_lower = entity_text.lower().strip()

        if entity_lower in GENERIC_ENTITIES or entity_lower in POSITION_TITLES:
            print(f"DEBUG - Original check: Skipping generic entity or title: {entity_text}")
            return True

        return False

    def _check_capitalized_word_after_entity(self, sentence: str, entity: Tuple[str, int, int]) -> bool:
        entity_text, start_pos, end_pos = entity

        text_after = sentence[end_pos:]
        if not text_after:
            return False

        import re
        pattern = r'^\s+([A-Z][a-z]+)'
        match = re.match(pattern, text_after)

        if match:
            capitalized_word = match.group(1)
            print(f"DEBUG - Found capitalized word immediately after entity: '{entity_text}' + '{capitalized_word}'")
            return True

        return False

    def _check_title_patterns_simple(self, sentence: str, entity: Tuple[str, int, int]) -> bool:
        if self.nlp is None:
            return False

        entity_text, start_pos, end_pos = entity

        try:
            context_start = max(0, start_pos - 50)
            context_end = min(len(sentence), end_pos + 50)
            context = sentence[context_start:context_end]

            doc = self.nlp(context)

            adj_start = start_pos - context_start
            adj_end = end_pos - context_start

            entity_tokens = [t for t in doc if adj_start <= t.idx < adj_end]

            for token in entity_tokens:
                if token.dep_ in ['compound', 'flat', 'amod', 'nmod']:
                    if (self._is_position_noun(token.head.text) or
                            any(self._is_position_noun(child.text) for child in token.children)):
                        print(f"DEBUG - spaCy simple: '{token.text}' is part of title structure")
                        return True

            return False

        except Exception as e:
            print(f"DEBUG - Simple title check failed: {e}")
            return False


    def _is_position_noun(self, word: str) -> bool:
        from rules.constants import POSITION_TITLES
        return word.lower() in POSITION_TITLES

    def _check_naming_verb_pattern(self, sentence: str, entity: Tuple[str, int, int]) -> bool:
        entity_text, start_pos, end_pos = entity
        text_before = sentence[:start_pos].strip()

        naming_verbs = {'called', 'named', 'nicknamed', 'known', 'said', 'referred', 'dubbed'}

        words_before = text_before.split()
        if words_before and words_before[-1].lower() in naming_verbs:
            print(f"DEBUG - Found naming verb '{words_before[-1]}' before '{entity_text}', skipping")
            return True

        return False


    def _check_dependency_syntax(self, sentence: str, entity: Tuple[str, int, int], entity_type: str) -> bool:
        if self.nlp is None:
            print("DEBUG - spaCy not available, skipping dependency analysis")
            return False

        entity_text, start_pos, end_pos = entity

        try:
            doc = self.nlp(sentence)

            entity_tokens = [t for t in doc if start_pos <= t.idx < end_pos]
            if not entity_tokens:
                print(f"DEBUG - spaCy: No tokens found for entity '{entity_text}'")
                return False

            head_token = self._find_entity_head(entity_tokens)

            print(
                f"DEBUG - spaCy analysis: '{entity_text}' head='{head_token.text}' dependency='{head_token.dep_}', pos='{head_token.pos_}'")

            if self._check_entity_level_dependencies(head_token, entity_tokens):
                return True

            if self._check_external_modifications(head_token, entity_tokens, doc):
                return True

            if self._check_quote_context(head_token, doc):
                return True

            print(f"DEBUG - spaCy: '{entity_text}' passed dependency analysis")
            return False

        except Exception as e:
            print(f"DEBUG - spaCy dependency analysis failed: {e}")
            return False

    def _find_entity_head(self, entity_tokens):
        if len(entity_tokens) == 1:
            return entity_tokens[0]

        for token in reversed(entity_tokens):  # 
            if token.dep_ in ['ROOT', 'nsubj', 'nsubjpass', 'dobj', 'pobj', 'conj']:
                return token

        return entity_tokens[-1]

    def _check_entity_level_dependencies(self, head_token, entity_tokens) -> bool:
        unsuitable_deps = {
            'appos',  
            'nmod',  
            'amod',  
            'det',  
            'poss'  
        }

        if head_token.dep_ in unsuitable_deps:
            print(f"DEBUG - spaCy: Entity-level unsuitable dependency '{head_token.dep_}', skipping")
            return True

        return False

    def _check_external_modifications(self, head_token, entity_tokens, doc) -> bool:
        entity_indices = {t.i for t in entity_tokens}

        for child in head_token.children:
            if child.i not in entity_indices:
                if child.dep_ in ['compound', 'amod', 'nmod'] and child.i < min(entity_indices):
                    print(f"DEBUG - spaCy: External modification by '{child.text}' ({child.dep_}), skipping")
                    return True

        entity_start_idx = min(entity_indices)
        if entity_start_idx > 0:
            prev_token = doc[entity_start_idx - 1]
            if prev_token.dep_ == 'amod' and prev_token.head.i in entity_indices:
                print(f"DEBUG - spaCy: Preceding adjective '{prev_token.text}', skipping")
                return True

        return False

    def _check_quote_context(self, head_token, doc) -> bool:
        start_idx = max(0, head_token.i - 2)
        end_idx = min(len(doc), head_token.i + 3)

        if any(t.is_quote for t in doc[start_idx:end_idx]):
            print(f"DEBUG - spaCy: In quotes context, skipping")
            return True
        return False

    def _check_problematic_preposition_pattern(self, sentence: str, entity: Tuple[str, int, int],
                                               entity_type: str) -> bool:
        entity_text, start_pos, end_pos = entity
        text_before = sentence[:start_pos].strip()
        words_before = text_before.split()

        if not words_before:
            return False

        last_word = words_before[-1].lower().rstrip('.,!?;:')

        if last_word == 'of':
            print(f"DEBUG - Found 'of' before '{entity_text}', 'of + pronoun' is grammatically problematic, skipping")
            return True

        return False


    def get_replacement_pronoun(self, entity_type: str, ner_model: str = "aws") -> str:
        if ner_model.lower() == "aws":
            type_mapping = {
                "PERSON": "he",
                "ORGANIZATION": "it",
                "LOCATION": "it",
                "COMMERCIAL_ITEM": "it",
                "TITLE": "it",
                "EVENT": "it",
                "OTHER": "it"
            }
        elif ner_model.lower() == "azure":
            type_mapping = {
                "Person": "he",  # 
                "PersonType": "they",  # 
                "Organization": "it",  # 
                "Location": "it",  # 
                "Event": "it",  # 
                "Product": "it"  # 
            }
        elif ner_model.lower() == "ontonotes":
            type_mapping = {
                "PERSON": "he",
                "ORG": "it",
                "GPE": "it",
                "LOC": "it",
                "FAC": "it",
                "EVENT": "it",
                "WORK_OF_ART": "it",
                "PRODUCT": "it",
                "NORP": "they",
                "LANGUAGE": "it",
                "LAW": "it"
            }
        elif ner_model.lower() == "conll3":
            type_mapping = {
                "PER": "he",  # 
                "ORG": "it",  # 
                "LOC": "it",  # 
                "MISC": "it"  # 
            }
        else:
            type_mapping = {
                "PERSON": "he",
                "ORGANIZATION": "it",
                "LOCATION": "it"
            }

        return type_mapping.get(entity_type, "it")

    def preview_replacement(self, sentence: str, entity: Tuple[str, int, int],
                            entity_type: str, ner_model: str = "aws") -> dict:
        entity_text, start_pos, end_pos = entity

        should_skip = self.should_skip_entity_replacement(sentence, entity, entity_type, ner_model)

        if should_skip:
            return {
                "original": sentence,
                "should_replace": False,
                "reason": "Entity should be skipped based on analysis",
                "ner_model": ner_model
            }
        else:
            pronoun = self.get_replacement_pronoun(entity_type, ner_model)
            replaced = sentence[:start_pos] + pronoun + sentence[end_pos:]
            return {
                "original": sentence,
                "replaced": replaced,
                "entity": entity_text,
                "pronoun": pronoun,
                "should_replace": True,
                "ner_model": ner_model
            }


comprehensive_checker = ComprehensiveEntityChecker()



def should_skip_entity_replacement_aws(sentence: str, entity: Tuple[str, int, int],
                                       entity_type: str = None) -> bool:
    """AWS """
    return comprehensive_checker.should_skip_entity_replacement(sentence, entity, entity_type, "aws")


def should_skip_entity_replacement_azure(sentence: str, entity: Tuple[str, int, int],
                                         entity_type: str = None) -> bool:
    """Azure """
    return comprehensive_checker.should_skip_entity_replacement(sentence, entity, entity_type, "azure")


def should_skip_entity_replacement_ontonotes(sentence: str, entity: Tuple[str, int, int],
                                             entity_type: str = None) -> bool:
    """OntoNotes """
    return comprehensive_checker.should_skip_entity_replacement(sentence, entity, entity_type, "ontonotes")


def should_skip_entity_replacement_comprehensive(sentence: str, entity: Tuple[str, int, int],
                                                 entity_type: str = None, ner_model: str = "aws") -> bool:
    return comprehensive_checker.should_skip_entity_replacement(sentence, entity, entity_type, ner_model)



def preview_entity_replacement_aws(sentence: str, entity: Tuple[str, int, int],
                                   entity_type: str = None) -> dict:
    """AWS """
    return comprehensive_checker.preview_replacement(sentence, entity, entity_type, "aws")


def preview_entity_replacement_azure(sentence: str, entity: Tuple[str, int, int],
                                     entity_type: str = None) -> dict:
    """Azure """
    return comprehensive_checker.preview_replacement(sentence, entity, entity_type, "azure")


def preview_entity_replacement_ontonotes(sentence: str, entity: Tuple[str, int, int],
                                         entity_type: str = None) -> dict:
    """OntoNotes """
    return comprehensive_checker.preview_replacement(sentence, entity, entity_type, "ontonotes")


def preview_entity_replacement(sentence: str, entity: Tuple[str, int, int],
                               entity_type: str = None, ner_model: str = "aws") -> dict:
    return comprehensive_checker.preview_replacement(sentence, entity, entity_type, ner_model)


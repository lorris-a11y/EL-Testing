from typing import Tuple

import spacy

# Import existing check functions directly
from rules.entity_skip_checks import (
   # has_pronoun_before_entity_spacy,
   is_entity_in_quotes,
   is_entity_in_formatted_brackets, has_spacy_noun_adjacent_to_ner_entity, has_pronoun_before_entity_spacy,
   has_definite_article_before_entity, has_possessive_before_entity, has_number_before_entity,
   has_ordinal_before_entity, has_wh_problem_before_entity_spacy
)
from .constants import GENERIC_ENTITIES, POSITION_TITLES


class ComprehensiveEntityChecker:
   def __init__(self):
       """Initialize spaCy model"""
       try:
           self.nlp = spacy.load('en_core_web_sm')
           print("DEBUG - spaCy model loaded successfully for comprehensive checking")
       except:
           self.nlp = None
           print("Warning: spaCy model not available")

   def should_skip_entity_replacement(self, sentence: str, entity: Tuple[str, int, int],
                                      entity_type: str = None, ner_model: str = "aws") -> bool:
       """
       Comprehensively determine if entity should skip pronoun replacement
       Supports multiple NER models

       Args:
           sentence: Sentence text
           entity: Entity tuple (text, start, end)
           entity_type: Entity type
           ner_model: NER model type ("aws", "azure", "ontonotes")
       """
       entity_text, start_pos, end_pos = entity

       print(f"DEBUG - Comprehensive check for '{entity_text}' ({entity_type}) [Model: {ner_model}]")

       # === Check 1: Original functionality - parentheses abbreviation check ===
       if self._check_abbreviation_in_parentheses(sentence, entity):
           return True

       # === Check 2: Original functionality - entity type filtering (adjusted by NER model) ===
       if self._check_non_replaceable_types(entity_type, ner_model):
           return True

       # === Check 3: Original functionality - generic word check ===
       if self._check_generic_entities(entity_text):
           return True

       # === Check 4: Quote check (directly use existing function) ===
       if is_entity_in_quotes(sentence, entity):
           print(f"DEBUG - Import check: Entity in quotes, skipping")
           return True

       # === Check 5: Formatted bracket check (directly use existing function) ===
       if is_entity_in_formatted_brackets(sentence, entity):
           print(f"DEBUG - Import check: Entity in formatted brackets, skipping")
           return True

       # === Check 6: Pronoun conflict check (directly use existing function) ===
       if has_pronoun_before_entity_spacy(sentence, entity):
           print(f"DEBUG - Import check: Pronoun before entity, skipping")
           return True

       # === Merged check: Noun before or after entity ===
       if has_spacy_noun_adjacent_to_ner_entity(sentence, entity):
           print(f"DEBUG - Import check: Noun adjacent to entity, skipping")
           return True

       # === New check: Number+entity pattern check ===
       if has_number_before_entity(sentence, entity):
           print(f"DEBUG - Import check: Number before entity, skipping")
           return True

       # === New check: Ordinal+entity pattern check ===
       if has_ordinal_before_entity(sentence, entity):
           print(f"DEBUG - Import check: Ordinal before entity, skipping")
           return True

       # === New check: spaCy precise detection of whose/those/these problems ===
       if has_wh_problem_before_entity_spacy(sentence, entity):
           print(f"DEBUG - spaCy check: Problematic pronoun before entity, skipping")
           return True

       # # === Check 7: Noun+entity check (directly use existing function) ===
       # if has_spacy_noun_before_ner_entity(sentence, entity):
       #     print(f"DEBUG - Import check: Noun before entity, skipping")
       #     return True

       # === Check 8: spaCy dependency syntax analysis ===
       if self._check_dependency_syntax(sentence, entity, entity_type):
           return True

       # === Check 9: Problematic preposition pattern ===
       if self._check_problematic_preposition_pattern(sentence, entity, entity_type):
           return True

       # === New check 10: Title pattern detection ===
       if self._check_title_patterns_simple(sentence, entity):
           return True

       # === New check: Entity overlap check (add these 4 lines) ===
       if self._check_entity_overlap_simple(sentence, entity):
           return True

       # === New check 10: Capitalized word after entity check ===
       if self._check_capitalized_word_after_entity(sentence, entity):
           print(f"DEBUG - Entity followed by capitalized word, skipping")
           return True

       # === New check 11: Definite article check ===
       if has_definite_article_before_entity(sentence, entity):
           print(f"DEBUG - Entity preceded by 'the', skipping to avoid 'the it' pattern")
           return True

       # === New check 12: Possessive check ===
       if has_possessive_before_entity(sentence, entity):
           print(f"DEBUG - Entity preceded by possessive, skipping to avoid \"'s it\" pattern")
           return True

       # === New check 13: Naming verb pattern ===
       if self._check_naming_verb_pattern(sentence, entity):
           return True

       print(f"DEBUG - All checks passed, allowing replacement for '{entity_text}'")
       return False


   # ==================== Original functionality checks ====================

   def _check_abbreviation_in_parentheses(self, sentence: str, entity: Tuple[str, int, int]) -> bool:
       """Original functionality 1: Check if entity is immediately followed by parentheses"""
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
       """Use spaCy to check if entity is part of a larger named entity"""
       if self.nlp is None:
           print("DEBUG - spaCy not available for entity overlap check")
           return False

       entity_text, start_pos, end_pos = entity

       try:
           doc = self.nlp(sentence)

           print(f"DEBUG - Checking overlap for '{entity_text}' at {start_pos}-{end_pos}")
           print(
               f"DEBUG - spaCy entities: {[(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}")

           # Iterate through all named entities identified by spaCy
           for ent in doc.ents:
               ent_start = ent.start_char
               ent_end = ent.end_char

               # Check for overlap but not exact match
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
       """Check entity types not suitable for replacement based on NER model"""
       if ner_model.lower() == "aws":
           non_replaceable_types = {"DATE", "QUANTITY"}
       elif ner_model.lower() == "azure":
           # Entity types in Azure NER not suitable for pronoun replacement
           non_replaceable_types = {
               "DateTime",  # Date time
               "PhoneNumber",  # Phone number
               "Email",  # Email
               "URL",  # URL
               "IP",  # IP address
               "Quantity",  # Quantity
               "Address",  # Address
               "Skill",   # Skill (usually abstract concept, not suitable for 'it' replacement)
               "PersonType",  # Person type (like job titles, usually not suitable for pronoun replacement)
           }
       elif ner_model.lower() == "ontonotes":
           non_replaceable_types = {
               "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY",
               "ORDINAL", "CARDINAL", "LANGUAGE"
           }
       elif ner_model.lower() == "conll3":
           # ConLL3 is four-class (PER, ORG, LOC, MISC), generally suitable for replacement
           # Unless there's special need, no filtering here
           non_replaceable_types = set()
       else:
           # Default case
           non_replaceable_types = {"DATE", "QUANTITY"}

       if entity_type in non_replaceable_types:
           print(f"DEBUG - {ner_model.upper()} check: Skipping non-replaceable entity type: {entity_type}")
           return True
       return False


   # Modified _check_generic_entities function
   def _check_generic_entities(self, entity_text: str) -> bool:
       entity_lower = entity_text.lower().strip()

       # Use imported constants instead of hardcoded lists
       if entity_lower in GENERIC_ENTITIES or entity_lower in POSITION_TITLES:
           print(f"DEBUG - Original check: Skipping generic entity or title: {entity_text}")
           return True

       return False

   def _check_capitalized_word_after_entity(self, sentence: str, entity: Tuple[str, int, int]) -> bool:
       """
       Check if entity is immediately followed by a capitalized word (only space-separated, no punctuation)
       """
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
       """Simplified title pattern detection"""
       if self.nlp is None:
           return False

       entity_text, start_pos, end_pos = entity

       try:
           # Only process text around entity for better performance
           context_start = max(0, start_pos - 50)
           context_end = min(len(sentence), end_pos + 50)
           context = sentence[context_start:context_end]

           doc = self.nlp(context)

           # Adjust coordinates
           adj_start = start_pos - context_start
           adj_end = end_pos - context_start

           entity_tokens = [t for t in doc if adj_start <= t.idx < adj_end]

           for token in entity_tokens:
               # Only check most important dependency relations
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
       """Check if it's a position noun"""
       position_words = {
           'minister', 'secretary', 'director', 'manager', 'officer', 'chief',
           'president', 'chairman', 'commissioner', 'administrator', 'deputy',
           'assistant', 'inspector', 'detective', 'sergeant', 'captain',
           'general', 'colonel', 'major', 'lieutenant', 'admiral', 'governor',
           'mayor', 'judge', 'justice', 'ambassador', 'consul', 'supervisor'
       }
       return word.lower() in position_words

   def _check_naming_verb_pattern(self, sentence: str, entity: Tuple[str, int, int]) -> bool:
       """Check if it's naming verb + entity pattern"""
       entity_text, start_pos, end_pos = entity
       text_before = sentence[:start_pos].strip()

       # Naming verb list
       naming_verbs = {'called', 'named', 'nicknamed', 'known', 'said', 'referred', 'dubbed'}

       words_before = text_before.split()
       if words_before and words_before[-1].lower() in naming_verbs:
           print(f"DEBUG - Found naming verb '{words_before[-1]}' before '{entity_text}', skipping")
           return True

       return False

   # ==================== spaCy dependency syntax analysis ====================

   def _check_dependency_syntax(self, sentence: str, entity: Tuple[str, int, int], entity_type: str) -> bool:
       """Simplified spaCy dependency syntax analysis"""
       if self.nlp is None:
           print("DEBUG - spaCy not available, skipping dependency analysis")
           return False

       entity_text, start_pos, end_pos = entity

       try:
           doc = self.nlp(sentence)

           # Find all tokens corresponding to entity
           entity_tokens = [t for t in doc if start_pos <= t.idx < end_pos]
           if not entity_tokens:
               print(f"DEBUG - spaCy: No tokens found for entity '{entity_text}'")
               return False

           # Find syntax head of entity
           head_token = self._find_entity_head(entity_tokens)

           print(
               f"DEBUG - spaCy analysis: '{entity_text}' head='{head_token.text}' dependency='{head_token.dep_}', pos='{head_token.pos_}'")

           # 1. Check entity-level dependencies
           if self._check_entity_level_dependencies(head_token, entity_tokens):
               return True

           # 2. Check external modifications of entity
           if self._check_external_modifications(head_token, entity_tokens, doc):
               return True

           # 3. Check quote context
           if self._check_quote_context(head_token, doc):
               return True

           print(f"DEBUG - spaCy: '{entity_text}' passed dependency analysis")
           return False

       except Exception as e:
           print(f"DEBUG - spaCy dependency analysis failed: {e}")
           return False

   def _find_entity_head(self, entity_tokens):
       """Find syntax head of entity"""
       if len(entity_tokens) == 1:
           return entity_tokens[0]

       # For multi-word entities, find the real head
       # Usually ROOT or token with dependency not compound/flat
       for token in reversed(entity_tokens):  # Search from back to front
           if token.dep_ in ['ROOT', 'nsubj', 'nsubjpass', 'dobj', 'pobj', 'conj']:
               return token

       # If no obvious head found, return last token
       return entity_tokens[-1]

   def _check_entity_level_dependencies(self, head_token, entity_tokens) -> bool:
       """Check entity-level dependencies"""
       # Entity-level dependencies not suitable for replacement
       unsuitable_deps = {
           'appos',  # Apposition: John, CEO of Microsoft
           'nmod',  # Modified by other nouns: the company Microsoft
           'amod',  # Adjective modification (if entity head)
           'det',  # Determiner
           'poss'  # Possessive
       }

       if head_token.dep_ in unsuitable_deps:
           print(f"DEBUG - spaCy: Entity-level unsuitable dependency '{head_token.dep_}', skipping")
           return True

       return False

   def _check_external_modifications(self, head_token, entity_tokens, doc) -> bool:
       """Check external modifications of entity"""
       entity_indices = {t.i for t in entity_tokens}

       # Check if head token's children have external modifiers
       for child in head_token.children:
           # Only consider modifiers outside entity
           if child.i not in entity_indices:
               if child.dep_ in ['compound', 'amod', 'nmod'] and child.i < min(entity_indices):
                   print(f"DEBUG - spaCy: External modification by '{child.text}' ({child.dep_}), skipping")
                   return True

       # Check if there are modifying words before entity
       entity_start_idx = min(entity_indices)
       if entity_start_idx > 0:
           prev_token = doc[entity_start_idx - 1]
           if prev_token.dep_ == 'amod' and prev_token.head.i in entity_indices:
               print(f"DEBUG - spaCy: Preceding adjective '{prev_token.text}', skipping")
               return True

       return False

   def _check_quote_context(self, head_token, doc) -> bool:
       """Check quote context"""
       # Check if there are quotes around entity
       start_idx = max(0, head_token.i - 2)
       end_idx = min(len(doc), head_token.i + 3)

       if any(t.is_quote for t in doc[start_idx:end_idx]):
           print(f"DEBUG - spaCy: In quotes context, skipping")
           return True
       return False

   # ==================== Check preposition + entity combination ====================
   def _check_problematic_preposition_pattern(self, sentence: str, entity: Tuple[str, int, int],
                                              entity_type: str) -> bool:
       """Check problematic preposition pattern - of + entity generally not suitable for pronoun replacement"""
       entity_text, start_pos, end_pos = entity
       text_before = sentence[:start_pos].strip()
       words_before = text_before.split()

       if not words_before:
           return False

       last_word = words_before[-1].lower().rstrip('.,!?;:')

       # 'of' followed by pronoun is generally grammatically problematic in English
       if last_word == 'of':
           print(f"DEBUG - Found 'of' before '{entity_text}', 'of + pronoun' is grammatically problematic, skipping")
           return True

       return False

   # ==================== Pronoun replacement logic ====================

   def get_replacement_pronoun(self, entity_type: str, ner_model: str = "aws") -> str:
       """Choose appropriate pronoun based on entity type and NER model"""
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
               "Person": "he",  # Person
               "PersonType": "they",  # Person type
               "Organization": "it",  # Organization
               "Location": "it",  # Location
               "Event": "it",  # Event
               "Product": "it"  # Product
               # Note: Types not suitable for replacement don't need pronoun definition here
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
               "PER": "he",  # Person
               "ORG": "it",  # Organization
               "LOC": "it",  # Location
               "MISC": "it"  # Other
           }
       else:
           # Default mapping
           type_mapping = {
               "PERSON": "she",
               "ORGANIZATION": "it",
               "LOCATION": "it"
           }

       return type_mapping.get(entity_type, "it")

   def preview_replacement(self, sentence: str, entity: Tuple[str, int, int],
                           entity_type: str, ner_model: str = "aws") -> dict:
       """Preview replacement effect"""
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


# Global instance
comprehensive_checker = ComprehensiveEntityChecker()


# ==================== Convenience functions for different NER models ====================

def should_skip_entity_replacement_aws(sentence: str, entity: Tuple[str, int, int],
                                      entity_type: str = None) -> bool:
   """AWS NER model specific"""
   return comprehensive_checker.should_skip_entity_replacement(sentence, entity, entity_type, "aws")


def should_skip_entity_replacement_azure(sentence: str, entity: Tuple[str, int, int],
                                        entity_type: str = None) -> bool:
   """Azure NER model specific"""
   return comprehensive_checker.should_skip_entity_replacement(sentence, entity, entity_type, "azure")


def should_skip_entity_replacement_ontonotes(sentence: str, entity: Tuple[str, int, int],
                                            entity_type: str = None) -> bool:
   """OntoNotes NER model specific"""
   return comprehensive_checker.should_skip_entity_replacement(sentence, entity, entity_type, "ontonotes")


def should_skip_entity_replacement_comprehensive(sentence: str, entity: Tuple[str, int, int],
                                                entity_type: str = None, ner_model: str = "aws") -> bool:
   """General function: comprehensively determine whether to skip entity replacement"""
   return comprehensive_checker.should_skip_entity_replacement(sentence, entity, entity_type, ner_model)


# ==================== Preview functions ====================

def preview_entity_replacement_aws(sentence: str, entity: Tuple[str, int, int],
                                  entity_type: str = None) -> dict:
   """AWS NER model preview"""
   return comprehensive_checker.preview_replacement(sentence, entity, entity_type, "aws")


def preview_entity_replacement_azure(sentence: str, entity: Tuple[str, int, int],
                                    entity_type: str = None) -> dict:
   """Azure NER model preview"""
   return comprehensive_checker.preview_replacement(sentence, entity, entity_type, "azure")


def preview_entity_replacement_ontonotes(sentence: str, entity: Tuple[str, int, int],
                                        entity_type: str = None) -> dict:
   """OntoNotes NER model preview"""
   return comprehensive_checker.preview_replacement(sentence, entity, entity_type, "ontonotes")


def preview_entity_replacement(sentence: str, entity: Tuple[str, int, int],
                              entity_type: str = None, ner_model: str = "aws") -> dict:
   """General preview function"""
   return comprehensive_checker.preview_replacement(sentence, entity, entity_type, ner_model)
"""
Helper functions module for entity replacement skip checks
Contains various auxiliary functions to check if entities should skip replacement
"""

import re
import spacy

from .constants import HONORIFICS_SET

# Load spaCy model
try:
   nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
   print("DEBUG - spaCy model loaded for entity skip checks")
except:
   nlp = None
   print("Warning: spaCy model 'en_core_web_sm' not loaded for entity skip checks.")

# Temporarily commented out, the function below includes this function's functionality, but unknown if it filters too much
def has_spacy_noun_before_ner_entity(sentence: str, entity: tuple) -> bool:
   """
   Check if spaCy identified nouns are before your NER model identified entities

   Args:
       sentence: Original sentence
       entity: (entity_text, start_pos, end_pos) entity identified by your NER model tagger

   Returns:
       bool: Returns True if spaCy identified nouns are before NER entity
   """
   entity_text, start_pos, end_pos = entity

   try:
       # Use spaCy to analyze the entire sentence for POS tagging
       doc = nlp(sentence)

       # Get text part before NER entity
       text_before_entity = sentence[:start_pos]

       if not text_before_entity.strip():
           return False

       # Find the last meaningful token in spaCy corresponding to text before NER entity
       last_meaningful_token = None

       for token in doc:
           # If token is completely within the text range before NER entity
           if token.idx + len(token.text) <= start_pos:
               # Skip punctuation and spaces
               if not token.is_punct and not token.is_space:
                   # Skip honorifics (avoid duplicate checks)
                   # honorifics_set = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'dame', 'lord', 'lady'}
                   # if token.text.lower().rstrip('.') not in honorifics_set:
                   #     last_meaningful_token = token
                   if token.text.lower().rstrip('.') not in HONORIFICS_SET:  # Use imported constant
                       last_meaningful_token = token

       # Check if the last meaningful token is a noun
       if last_meaningful_token and last_meaningful_token.pos_ in ['NOUN', 'PROPN']:
           print(
               f"DEBUG - spaCy found {last_meaningful_token.pos_} '{last_meaningful_token.text}' before NER entity '{entity_text}' (identified by your tagger)")
           return True

       return False

   except Exception as e:
       print(f"DEBUG - spaCy analysis failed: {e}")
       return False


def has_pronoun_before_entity_spacy(sentence: str, entity: tuple) -> bool:
   """
   Use spaCy to check if there's a pronoun before NER entity
   Prevent generating incorrect "pronoun+pronoun" patterns

   Args:
       sentence: Original sentence
       entity: (entity_text, start_pos, end_pos) entity identified by NER model

   Returns:
       bool: Returns True if there's a pronoun before NER entity (avoid pronoun+pronoun after replacement)
   """
   entity_text, start_pos, end_pos = entity

   try:
       # Use spaCy to analyze the entire sentence
       doc = nlp(sentence)

       # Get text part before NER entity
       text_before_entity = sentence[:start_pos]

       if not text_before_entity.strip():
           return False

       # Find the last meaningful token before NER entity
       last_meaningful_token = None

       for token in doc:
           # If token is completely within the text range before NER entity
           if token.idx + len(token.text) <= start_pos:
               # Skip punctuation and spaces
               if not token.is_punct and not token.is_space:
                   last_meaningful_token = token

       # Check if the last meaningful token is a pronoun
       if last_meaningful_token and last_meaningful_token.pos_ == 'PRON':
           print(
               f"DEBUG - spaCy found pronoun '{last_meaningful_token.text}' before NER entity '{entity_text}' - skipping to avoid pronoun+pronoun pattern")
           return True

       return False

   except Exception as e:
       print(f"DEBUG - spaCy pronoun analysis failed: {e}")
       return False

# Consider both noun+entity and entity+noun situations
def has_spacy_noun_adjacent_to_ner_entity(sentence: str, entity: tuple) -> bool:
   """
   Check if spaCy identified nouns are before or after your NER model identified entities

   Args:
       sentence: Original sentence
       entity: (entity_text, start_pos, end_pos) entity identified by your NER model tagger

   Returns:
       bool: Returns True if spaCy identified nouns are before or after NER entity
   """
   entity_text, start_pos, end_pos = entity

   try:
       # Use spaCy to analyze the entire sentence for POS tagging
       doc = nlp(sentence)

       # Check if there's a noun before the entity
       text_before_entity = sentence[:start_pos]
       if text_before_entity.strip():
           # Find the last meaningful token in spaCy corresponding to text before NER entity
           last_meaningful_token = None

           for token in doc:
               # If token is completely within the text range before NER entity
               if token.idx + len(token.text) <= start_pos:
                   # Skip punctuation and spaces
                   if not token.is_punct and not token.is_space:
                       # Skip honorifics (avoid duplicate checks)
                       honorifics_set = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'dame', 'lord', 'lady'}
                       if token.text.lower().rstrip('.') not in honorifics_set:
                           last_meaningful_token = token

           # Check if the last meaningful token is a noun
           if last_meaningful_token and last_meaningful_token.pos_ in ['NOUN', 'PROPN']:
               print(
                   f"DEBUG - spaCy found {last_meaningful_token.pos_} '{last_meaningful_token.text}' before NER entity '{entity_text}'")
               return True

       # Check if there's a noun after the entity
       text_after_entity = sentence[end_pos:]
       if text_after_entity.strip():
           # Find the first meaningful token in spaCy corresponding to text after NER entity
           first_meaningful_token = None

           for token in doc:
               # If token is completely within the text range after NER entity
               if token.idx >= end_pos:
                   # Skip punctuation and spaces
                   if not token.is_punct and not token.is_space:
                       first_meaningful_token = token
                       break

           # Check if the first meaningful token is a common noun
           if (first_meaningful_token and
                   first_meaningful_token.pos_ == 'NOUN' and  # Is common noun, not proper noun
                   not first_meaningful_token.ent_type_):  # Not part of a named entity

               print(
                   f"DEBUG - spaCy found NOUN '{first_meaningful_token.text}' after NER entity '{entity_text}'")
               return True

       return False

   except Exception as e:
       print(f"DEBUG - spaCy noun adjacency analysis failed: {e}")
       return False


def is_entity_in_quotes(sentence: str, entity: tuple) -> bool:
   """
   Check if entity is within quotes

   Args:
       sentence: Original sentence
       entity: (entity_text, start_pos, end_pos) tuple

   Returns:
       bool: Returns True if entity is within quotes
   """
   entity_text, start_pos, end_pos = entity

   # Check if there are quotes before and after the entity
   text_before = sentence[:start_pos]
   text_after = sentence[end_pos:]

   # Check various quote types
   quote_chars = ['"', "'", '"', '"', ''', ''']

   for quote in quote_chars:
       # Check if surrounded by quotes
       if quote in text_before and quote in text_after:
           # Find the nearest quote before entity
           last_quote_before = text_before.rfind(quote)
           # Find the nearest quote after entity
           first_quote_after = text_after.find(quote)

           if last_quote_before != -1 and first_quote_after != -1:
               # Check if the content between quotes mainly contains this entity
               quoted_content = sentence[last_quote_before + 1:end_pos + first_quote_after]
               if entity_text in quoted_content and len(quoted_content.strip()) <= len(entity_text) + 10:
                   print(f"DEBUG - Skipping entity in quotes: '{entity_text}' appears to be quoted")
                   return True

   return False


def is_entity_in_formatted_brackets(sentence: str, entity: tuple) -> bool:
   """
   Check if entity is within formatted brackets (like time format, technical format, etc.)

   Args:
       sentence: Original sentence
       entity: (entity_text, start_pos, end_pos) tuple

   Returns:
       bool: Returns True if entity is within formatted brackets
   """
   entity_text, start_pos, end_pos = entity

   # Check if entity is within brackets
   text_before = sentence[:start_pos]
   text_after = sentence[end_pos:]

   # Find the nearest opening bracket
   last_open_bracket = text_before.rfind('(')
   # Find the nearest closing bracket
   first_close_bracket = text_after.find(')')

   if last_open_bracket != -1 and first_close_bracket != -1:
       # Extract content within brackets
       bracket_content = sentence[last_open_bracket + 1:end_pos + first_close_bracket]

       # Check if it's formatted content patterns
       format_patterns = [
           # Time format: (03:47 CET/02:47 GMT)
           r'\d{1,2}:\d{2}\s+\w+/\d{1,2}:\d{2}\s+\w+',
           # Date format: (2024 AD/1445 AH)
           r'\d{4}\s+\w+/\d{4}\s+\w+',
           # Technical format: (v1.2.3/beta)
           r'v?\d+\.\d+[\.\d]*\s*/\s*\w+',
           # Currency format: ($100/€85)
           r'[\$€£¥]\d+/[\$€£¥]\d+',
           # Simple separated format: short format containing "/"
           r'^[^/]{1,15}/[^/]{1,15}$'
       ]

       for pattern in format_patterns:
           if re.search(pattern, bracket_content):
               print(f"DEBUG - Skipping entity in formatted brackets: '{entity_text}' in '{bracket_content}'")
               return True

       # Additional check: if bracket content is short and contains slash, might be formatted content
       if '/' in bracket_content and len(bracket_content.strip()) < 30:
           # Check if mainly composed of letters, digits, slash, colon, space
           if re.match(r'^[\w\s/:.-]+$', bracket_content.strip()):
               print(f"DEBUG - Skipping entity in likely formatted brackets: '{entity_text}' in '{bracket_content}'")
               return True

   return False


def has_capitalized_word_after_entity(sentence: str, entity: tuple) -> bool:
   """
   Check if entity is immediately followed by a capitalized word (only space-separated, no punctuation)

   Args:
       sentence: Complete sentence
       entity: Entity tuple (entity_text, start_pos, end_pos)

   Returns:
       bool: Returns True if entity is immediately followed by a capitalized word
   """
   entity_text, start_pos, end_pos = entity

   # Get text after entity, but don't remove leading spaces (need to check if only spaces)
   text_after = sentence[end_pos:]

   if not text_after:
       return False

   # Check if after entity there are only spaces, then a capitalized word
   # Use regex for exact matching: only spaces + capitalized word
   import re

   # Pattern: leading spaces + capitalized word with lowercase letters following
   pattern = r'^\s+([A-Z][a-z]+)'
   match = re.match(pattern, text_after)

   if match:
       capitalized_word = match.group(1)
       print(f"DEBUG - Found capitalized word immediately after entity: '{entity_text}' + '{capitalized_word}'")
       return True

   return False


def has_definite_article_before_entity(sentence: str, entity: tuple) -> bool:
   """
   Check if there's definite article 'the' before entity
   Avoid generating unreasonable structures like "the it"

   Args:
       sentence: Original sentence
       entity: (entity_text, start_pos, end_pos) tuple

   Returns:
       bool: Returns True if there's definite article before entity
   """
   entity_text, start_pos, end_pos = entity

   # Get text before entity
   text_before = sentence[:start_pos].rstrip()

   if not text_before:
       return False

   # Check if ends with "the" (case insensitive)
   words_before = text_before.split()
   if words_before:
       last_word = words_before[-1].lower()
       if last_word == "the":
           print(f"DEBUG - Found definite article 'the' before entity '{entity_text}'")
           return True

   return False


def has_possessive_before_entity(sentence: str, entity: tuple) -> bool:
   """
   Check if there's possessive form before entity (like Elon Musk's, government's etc.)
   Avoid generating unreasonable structures like "Elon Musk's it"
   """
   entity_text, start_pos, end_pos = entity

   # Get text before entity
   text_before = sentence[:start_pos].rstrip()

   if not text_before:
       return False

   # Simple text pattern check
   possessive_patterns = [
       r"'s\s*$",  # Regular possessive: John's, company's
       r"s'\s*$",  # Plural possessive: companies', students'
       r"'\s*$",  # Special cases: Jesus', boss'
   ]

   import re
   for pattern in possessive_patterns:
       if re.search(pattern, text_before):
           print(f"DEBUG - Found possessive pattern before entity '{entity_text}'")
           return True

   # Use spaCy for more precise detection
   if nlp:
       try:
           doc = nlp(sentence)

           # Find tokens corresponding to entity position
           entity_tokens = []
           for token in doc:
               if start_pos <= token.idx < end_pos:
                   entity_tokens.append(token)

           if entity_tokens:
               first_entity_token = entity_tokens[0]

               # Check if entity's first token has possessive dependency
               if first_entity_token.dep_ == "poss":
                   print(f"DEBUG - spaCy detected entity '{entity_text}' has possessive dependency")
                   return True

               # Check if previous token is possessive marker
               if first_entity_token.i > 0:
                   prev_token = doc[first_entity_token.i - 1]
                   if prev_token.tag_ == "POS" or prev_token.text in ["'s", "'", "s'"]:
                       print(
                           f"DEBUG - spaCy detected possessive marker '{prev_token.text}' before entity '{entity_text}'")
                       return True

       except Exception as e:
           print(f"DEBUG - spaCy possessive analysis failed: {e}")
           # If spaCy fails, rely on previous pattern matching results

   return False


def has_number_before_entity(sentence: str, entity: tuple) -> bool:
   """
   Check if there's a number or quantifier before entity, avoid generating unreasonable structures like "700 it"

   Args:
       sentence: Original sentence
       entity: (entity_text, start_pos, end_pos) tuple

   Returns:
       bool: Returns True if there's a number before entity
   """
   entity_text, start_pos, end_pos = entity

   # Get text before entity
   text_before = sentence[:start_pos].strip()

   if not text_before:
       return False

   import re

   # Simplified method: check if text ends with number (possibly with preceding modifiers)
   number_pattern = r'\b\d+(?:\.\d+)?(?:\s*%)?$'  # Match text ending with number (including decimals and percentages)

   if re.search(number_pattern, text_before):
       print(f"DEBUG - Found number+entity pattern: '{text_before}' + '{entity_text}', skipping")
       return True

   # Check large numbers with commas (like 1,000)
   comma_number_pattern = r'\b\d{1,3}(?:,\d{3})+$'
   if re.search(comma_number_pattern, text_before):
       print(f"DEBUG - Found comma-separated number+entity pattern: '{text_before}' + '{entity_text}', skipping")
       return True

   # Check pure quantifier expressions (without specific numbers)
   quantifier_only_patterns = [
       r'\b(?:several|many|few|countless|innumerable|numerous)\s*$',
       r'\b(?:dozens?|hundreds?|thousands?|millions?)\s+of\s*$',
       r'\ba\s+(?:couple|few|dozen|hundred|thousand|million)\s+(?:of\s+)?$',
       r'\b(?:some|multiple)\s*$',
   ]

   for pattern in quantifier_only_patterns:
       if re.search(pattern, text_before, re.IGNORECASE):
           print(f"DEBUG - Found quantifier+entity pattern: '{text_before}' + '{entity_text}', skipping")
           return True

   # Use spaCy for more precise detection
   if nlp:
       try:
           # Only analyze text before entity for efficiency
           doc = nlp(text_before)

           # Check if last few tokens contain numbers
           tokens = [t for t in doc if not t.is_punct and not t.is_space]
           if tokens:
               # Check last 1-2 meaningful tokens
               for token in tokens[-2:]:
                   # spaCy number detection
                   if token.pos_ == "NUM" or token.like_num:
                       print(f"DEBUG - spaCy detected number token '{token.text}' before entity '{entity_text}'")
                       return True

                   # Check if it's quantity-related word
                   if token.lemma_.lower() in {"dozen", "hundred", "thousand", "million", "billion",
                                               "many", "few", "several", "some", "multiple"}:
                       print(f"DEBUG - spaCy detected quantifier '{token.text}' before entity '{entity_text}'")
                       return True

       except Exception as e:
           print(f"DEBUG - spaCy number analysis failed: {e}")

   return False


def has_ordinal_before_entity(sentence: str, entity: tuple) -> bool:
   """
   Check if there's an ordinal number before entity, avoid generating unreasonable structures like "first it"

   Args:
       sentence: Original sentence
       entity: (entity_text, start_pos, end_pos) tuple

   Returns:
       bool: Returns True if there's an ordinal number before entity
   """
   entity_text, start_pos, end_pos = entity

   # Get text before entity
   text_before = sentence[:start_pos].strip()

   if not text_before:
       return False

   import re

   # Ordinal patterns
   ordinal_patterns = [
       r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)$',
       r'\b(?:eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)$',
       r'\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)$',
       r'\b\d+(?:st|nd|rd|th)$',  # 1st, 2nd, 3rd, 4th etc
       r'\bthe\s+(?:first|second|third|last|final|initial|primary)$',
   ]

   for pattern in ordinal_patterns:
       if re.search(pattern, text_before, re.IGNORECASE):
           print(f"DEBUG - Found ordinal+entity pattern: '{text_before}' + '{entity_text}', skipping")
           return True

   return False


def has_wh_problem_before_entity_spacy(sentence: str, entity: tuple) -> bool:
   """
   Args:
       sentence: Original sentence
       entity: (entity_text, start_pos, end_pos) tuple

   Returns:
       bool: Returns True if there's problematic pronoun before entity
   """
   if nlp is None:
       print("DEBUG - spaCy not available for pronoun checking")
       return False

   entity_text, start_pos, end_pos = entity

   try:
       # Analyze entire sentence
       doc = nlp(sentence)

       # Find first token corresponding to entity
       entity_start_token = None
       for token in doc:
           if start_pos <= token.idx < end_pos:
               entity_start_token = token
               break

       if not entity_start_token or entity_start_token.i == 0:
           return False

       # Find meaningful token before entity
       preceding_token = None
       for i in range(entity_start_token.i - 1, -1, -1):
           token = doc[i]
           if not token.is_punct and not token.is_space and token.text.strip():
               preceding_token = token
               break

       if not preceding_token:
           return False

       print(
           f"DEBUG - spaCy pronoun check: '{preceding_token.text}' (POS: {preceding_token.pos_}, TAG: {preceding_token.tag_}) + '{entity_text}'")

       # === 1. Check possessive interrogative pronoun "whose" ===
       if preceding_token.tag_ == "WP$":  # Wh-possessive
           print(f"DEBUG - spaCy: Found possessive wh-word '{preceding_token.text}' before '{entity_text}', skipping")
           return True

       # === 2. Check demonstrative pronouns "those/these" ===
       # spaCy usually tags these/those as DET(determiner), but in some contexts might be PRON
       if (preceding_token.text.lower() in ["these", "those"] and
               preceding_token.pos_ in ["DET", "PRON"]):
           print(f"DEBUG - spaCy: Found demonstrative '{preceding_token.text}' before '{entity_text}', skipping")
           return True

       # === 3. Additional check: "which" in certain contexts ===
       # In some cases "which company" → "which it" might also be problematic
       if (preceding_token.text.lower() == "which" and
               preceding_token.tag_ in ["WDT", "WP"]):  # Wh-determiner or Wh-pronoun
           # Further check context to determine if it's really problematic
           print(f"DEBUG - spaCy: Found 'which' before '{entity_text}', may need skipping")
           return True

       return False

   except Exception as e:
       print(f"DEBUG - spaCy pronoun analysis failed: {e}")
       return False
"""
entity_swap_filters.py
Module for filtering entity types that are not suitable for swapping between sentences
Supports four NER models: ConLL3, OntoNotes, Azure NER, AWS NER
"""

from typing import Dict, Set, Tuple, List
import re


class EntitySwapFilter:
   """Entity swap filter for determining which entities are not suitable for swapping between sentences"""

   def __init__(self):
       # Define entity types that are not suitable for swapping in each NER model
       self.non_swappable_types = {
           "conll3": {
               # ConLL3 is relatively simple, most can be swapped
           },

           "ontonotes": {
               # Numerical and time-related
               "DATE",  # Date
               "TIME",  # Time
               "PERCENT",  # Percentage
               "MONEY",  # Money
               "QUANTITY",  # Quantity
               "ORDINAL",  # Ordinal numbers
               "CARDINAL",  # Cardinal numbers
               "LANGUAGE",  # Language names
           },

           "azure": {
               # Contact information and identifiers
               "DateTime",  # Date time
               "PhoneNumber",  # Phone number
               "Email",  # Email
               "URL",  # URL
               "IP",  # IP address
               "Address",  # Address
               "Quantity",  # Quantity
           },

           "aws": {
               # Numerical and time-related
               "DATE",  # Date
               "QUANTITY",  # Quantity
           }
       }

   def should_filter_entity(self, entity: Dict, sentence: str, ner_model: str) -> Tuple[bool, str]:
       """
       Determine if entity should be filtered (not swapped)

       Args:
           entity: Entity dictionary containing text, tag, start, end
           sentence: Sentence containing the entity
           ner_model: NER model name

       Returns:
           (should_filter, reason): Whether to filter and reason
       """
       entity_text = entity["text"]
       entity_type = entity["tag"]
       start_pos = entity["start"]
       end_pos = entity["end"]

       ner_model = ner_model.lower()

       # 1. Check if it's a non-swappable type
       if ner_model in self.non_swappable_types:
           if entity_type in self.non_swappable_types[ner_model]:
               return True, f"Entity type '{entity_type}' is non-swappable in {ner_model}"

       # 2. Check if it contains special characters
       if self._contains_special_characters(entity_text):
           return True, f"Entity contains special characters: '{entity_text}'"

       # 3. Check if it's an abbreviation
       if self._is_abbreviation(entity_text):
           return True, f"Entity is abbreviation: '{entity_text}'"

       # 4. Check if it's in parentheses
       if self._is_in_parentheses(sentence, start_pos, end_pos):
           return True, f"Entity is in parentheses: '{entity_text}'"

       # 5. Check if it's in quotes
       if self._is_in_quotes(sentence, start_pos, end_pos):
           return True, f"Entity is in quotes: '{entity_text}'"

       return False, "Entity is swappable"

   def _contains_special_characters(self, entity_text: str) -> bool:
       """Check if contains special characters"""
       # Allowed characters: letters, digits, spaces, hyphens, apostrophes, dots
       allowed_pattern = r'^[a-zA-Z0-9\s\-\'\.]+$'
       return not re.match(allowed_pattern, entity_text)

   def _is_abbreviation(self, entity_text: str) -> bool:
       """Check if it's an abbreviation"""
       # All uppercase and length <= 5
       if entity_text.isupper() and len(entity_text) <= 5:
           return True
       # Contains multiple dots (like U.S.A.)
       if entity_text.count('.') >= 2:
           return True
       # Single uppercase letter followed by dot (like J. in J. Smith)
       if re.match(r'^[A-Z]\.$', entity_text):
           return True
       return False

   def _is_in_parentheses(self, sentence: str, start_pos: int, end_pos: int) -> bool:
       """Check if entity is in parentheses"""
       # Check if there's opening parenthesis before entity
       before_text = sentence[:start_pos].rstrip()
       after_text = sentence[end_pos:].lstrip()

       # Case 1: Entity is surrounded by parentheses
       if before_text.endswith('(') and after_text.startswith(')'):
           return True

       # Case 2: Entity is inside parentheses (find nearest parentheses)
       # Search backward for nearest parenthesis
       open_paren = before_text.rfind('(')
       close_paren_before = before_text.rfind(')')

       # Search forward for nearest parenthesis
       close_paren_after = after_text.find(')')

       # If there's unpaired opening parenthesis and closing parenthesis after
       if open_paren != -1 and close_paren_after != -1:
           # Ensure no paired closing parenthesis after opening parenthesis
           if close_paren_before < open_paren:
               return True

       return False

   def _is_in_quotes(self, sentence: str, start_pos: int, end_pos: int) -> bool:
       """Check if entity is in quotes"""
       # Get text before and after entity
       before_text = sentence[:start_pos]
       after_text = sentence[end_pos:]

       # Check various quote types
       quote_pairs = [
           ('"', '"'),  # Double quotes
           ("'", "'"),  # Single quotes
           ('"', '"'),  # Chinese/smart quotes
           (''', '''),  # Chinese/smart single quotes
           ('「', '」'),  # Japanese quotes
           ('『', '』'),  # Japanese quotes
           ('«', '»'),  # French quotes
           ('‹', '›'),  # French single quotes
       ]

       for open_quote, close_quote in quote_pairs:
           # Count opening and closing quotes before entity
           open_count_before = before_text.count(open_quote)
           close_count_before = before_text.count(close_quote)

           # Count closing quotes after entity
           close_count_after = after_text.count(close_quote)

           # If more opening quotes than closing quotes before entity, and closing quotes after, entity is inside quotes
           if open_count_before > close_count_before and close_count_after > 0:
               return True

           # Special case: directly check if entity is surrounded by quotes
           # For example: "Green Coffin" or 'iSpoof'
           if start_pos > 0 and end_pos < len(sentence):
               if (sentence[start_pos - 1] == open_quote and
                       sentence[end_pos] == close_quote):
                   return True

               # Check if it's \"entity\" format (escaped quotes)
               if (start_pos >= 2 and
                       sentence[start_pos - 2:start_pos] == '\\' + open_quote and
                       end_pos + 1 < len(sentence) and
                       sentence[end_pos:end_pos + 2] == close_quote + '\\'):
                   return True

               # Check if it's \"entity\" format, but no backslash after quote
               if (start_pos >= 2 and
                       sentence[start_pos - 2:start_pos] == '\\' + open_quote and
                       end_pos < len(sentence) and
                       sentence[end_pos] == close_quote):
                   return True

       # Additional check: use regex to find all quote pairs
       # This can handle complex cases like nested quotes
       patterns = [
           r'"[^"]*"',  # Double quote content
           r"'[^']*'",  # Single quote content
           r'\\"[^"]*\\"',  # Escaped double quote content
       ]

       for pattern in patterns:
           for match in re.finditer(pattern, sentence):
               match_start = match.start()
               match_end = match.end()
               # Check if entity is within matched quotes
               if match_start < start_pos and end_pos < match_end:
                   return True

       return False


# Create global filter instance
entity_swap_filter = EntitySwapFilter()


# Convenience function
def should_filter_entity_for_swap(entity: Dict, sentence: str, ner_model: str) -> Tuple[bool, str]:
   """Determine if entity should be filtered"""
   return entity_swap_filter.should_filter_entity(entity, sentence, ner_model)
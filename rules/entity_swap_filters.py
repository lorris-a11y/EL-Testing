
from typing import Dict, Set, Tuple, List
import re


class EntitySwapFilter:

    def __init__(self):
        # 定义各个NER模型中不适合交换的实体类型
        self.non_swappable_types = {
            "conll3": {
            },

            "ontonotes": {
                "DATE",  # 
                "TIME",  # 
                "PERCENT",  # 
                "MONEY",  # 
                "QUANTITY",  # 
                "ORDINAL",  # 
                "CARDINAL",  # 
                "LANGUAGE",  # 
            },

            "azure": {
                "DateTime",  # 
                "PhoneNumber",  # 
                "Email",  # 
                "URL",  # 
                "IP",  # 
                "Address",  # 
                "Quantity",  # 
            },

            "aws": {
                "DATE",  # 
                "QUANTITY",  # 
            }
        }

    def should_filter_entity(self, entity: Dict, sentence: str, ner_model: str) -> Tuple[bool, str]:

        entity_text = entity["text"]
        entity_type = entity["tag"]
        start_pos = entity["start"]
        end_pos = entity["end"]

        ner_model = ner_model.lower()

        if ner_model in self.non_swappable_types:
            if entity_type in self.non_swappable_types[ner_model]:
                return True, f"Entity type '{entity_type}' is non-swappable in {ner_model}"

        if self._contains_special_characters(entity_text):
            return True, f"Entity contains special characters: '{entity_text}'"

        if self._is_abbreviation(entity_text):
            return True, f"Entity is abbreviation: '{entity_text}'"

        if self._is_in_parentheses(sentence, start_pos, end_pos):
            return True, f"Entity is in parentheses: '{entity_text}'"

        if self._is_in_quotes(sentence, start_pos, end_pos):
            return True, f"Entity is in quotes: '{entity_text}'"

        return False, "Entity is swappable"

    def _contains_special_characters(self, entity_text: str) -> bool:
        allowed_pattern = r'^[a-zA-Z0-9\s\-\'\.]+$'
        return not re.match(allowed_pattern, entity_text)

    def _is_abbreviation(self, entity_text: str) -> bool:

        if entity_text.isupper() and len(entity_text) <= 5:
            return True
        if entity_text.count('.') >= 2:
            return True
        if re.match(r'^[A-Z]\.$', entity_text):
            return True
        return False

    def _is_in_parentheses(self, sentence: str, start_pos: int, end_pos: int) -> bool:

        before_text = sentence[:start_pos].rstrip()
        after_text = sentence[end_pos:].lstrip()

        if before_text.endswith('(') and after_text.startswith(')'):
            return True

        open_paren = before_text.rfind('(')
        close_paren_before = before_text.rfind(')')

        close_paren_after = after_text.find(')')

        if open_paren != -1 and close_paren_after != -1:
            if close_paren_before < open_paren:
                return True

        return False

    def _is_in_quotes(self, sentence: str, start_pos: int, end_pos: int) -> bool:
        before_text = sentence[:start_pos]
        after_text = sentence[end_pos:]

        quote_pairs = [
            ('"', '"'),  
            ("'", "'"),  
            ('"', '"'),  
            (''', '''),  
            ('「', '」'),  
            ('『', '』'),  
            ('«', '»'),  
            ('‹', '›'),  
        ]

        for open_quote, close_quote in quote_pairs:
            open_count_before = before_text.count(open_quote)
            close_count_before = before_text.count(close_quote)

            close_count_after = after_text.count(close_quote)

            if open_count_before > close_count_before and close_count_after > 0:
                return True


            if start_pos > 0 and end_pos < len(sentence):
                if (sentence[start_pos - 1] == open_quote and
                        sentence[end_pos] == close_quote):
                    return True

                if (start_pos >= 2 and
                        sentence[start_pos - 2:start_pos] == '\\' + open_quote and
                        end_pos + 1 < len(sentence) and
                        sentence[end_pos:end_pos + 2] == close_quote + '\\'):
                    return True

                if (start_pos >= 2 and
                        sentence[start_pos - 2:start_pos] == '\\' + open_quote and
                        end_pos < len(sentence) and
                        sentence[end_pos] == close_quote):
                    return True


        patterns = [
            r'"[^"]*"',  # 
            r"'[^']*'",  # 
            r'\\"[^"]*\\"',  # 
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, sentence):
                match_start = match.start()
                match_end = match.end()
                if match_start < start_pos and end_pos < match_end:
                    return True

        return False


entity_swap_filter = EntitySwapFilter()


def should_filter_entity_for_swap(entity: Dict, sentence: str, ner_model: str) -> Tuple[bool, str]:
    """判断实体是否应该被过滤"""
    return entity_swap_filter.should_filter_entity(entity, sentence, ner_model)
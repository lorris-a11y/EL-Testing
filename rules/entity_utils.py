# entity_utils.py
"""
共享的实体处理工具函数
这些函数被所有模块使用，不依赖其他自定义模块
"""

import re
from typing import Tuple

from rules.constants import PREPOSITIONS


class EntitySpan:
    """Class to hold entity span information"""

    def __init__(self, start: int, end: int, honorific: str = None, article: str = None):
        self.start = start
        self.end = end
        self.honorific = honorific
        self.article = article


# def find_full_entity_span(sentence: str, entity: tuple) -> EntitySpan:
#     """找到实体的完整范围，包括敬称和定冠词"""
#     entity_text, start_pos, end_pos = entity
#     prefix_text = sentence[:start_pos].rstrip()
#
#     span = EntitySpan(start_pos, end_pos)
#
#     if prefix_text.endswith("the ") or prefix_text.endswith("The "):
#         span.article = "the"
#         span.start = start_pos - 4
#
#     honorifics = ["Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.", "Dr", "Dr.", "Prof", "Prof."]
#     for honorific in honorifics:
#         if prefix_text.endswith(honorific + " "):
#             span.honorific = honorific
#             span.start = start_pos - len(honorific) - 1
#             break
#
#     return span

def find_full_entity_span(sentence: str, entity: tuple) -> EntitySpan:
    """找到实体的完整范围，包括敬称和定冠词"""
    from rules.constants import HONORIFICS

    entity_text, start_pos, end_pos = entity
    prefix_text = sentence[:start_pos].rstrip()

    span = EntitySpan(start_pos, end_pos)

    if prefix_text.endswith("the ") or prefix_text.endswith("The "):
        span.article = "the"
        span.start = start_pos - 4

    # ✅ 使用 constants 中的定义
    for honorific in HONORIFICS:
        if prefix_text.endswith(honorific + " "):
            span.honorific = honorific
            span.start = start_pos - len(honorific) - 1
            break

    return span


def adjust_sentence_structure(sentence: str) -> str:
    """调整句子结构，确保语法正确"""
    sentence = re.sub(r'\s+', ' ', sentence)

    fixes = [
        (r'Mr\.?\s+he', 'he'),
        (r'Mrs\.?\s+she', 'she'),
        (r'Ms\.?\s+she', 'she'),
        (r'Dr\.?\s+he', 'he'),
        (r'Prof\.?\s+he', 'he'),
        (r'Mr\.?\s+his', 'his'),
        (r'Mr\.?\s+him', 'him'),
        (r'Mrs\.?\s+her', 'her'),
        (r'Ms\.?\s+her', 'her'),
        (r'Dr\.?\s+his', 'his'),
        (r'Dr\.?\s+him', 'him'),
        (r'Prof\.?\s+his', 'his'),
        (r'Prof\.?\s+him', 'him'),
        (r'[Tt]he\s+it\b(?!\s+is)', ' it'),
        (r'[Tt]he\s+its\b', ' its'),
        (r'[Tt]he\s+he\b', ' he'),
        (r'[Tt]he\s+him\b', ' him'),
        (r'[Tt]he\s+his\b', ' his'),
        (r'[Tt]he\s+she\b', ' she'),
        (r'[Tt]he\s+her\b', ' her'),
        (r'(?:^|\s)(?:a|an)\s+it\b(?!\s+is)', ' it'),
        (r'(?:^|\s)(?:a|an)\s+its\b', ' its'),
        (r'(?:^|\s)(?:a|an)\s+he\b', ' he'),
        (r'(?:^|\s)(?:a|an)\s+him\b', ' him'),
        (r'(?:^|\s)(?:a|an)\s+his\b', ' his'),
        (r'(?:^|\s)(?:a|an)\s+she\b', ' she'),
        (r'(?:^|\s)(?:a|an)\s+her\b', ' her'),
    ]

    result = sentence
    for pattern, replacement in fixes:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # prepositions = ['at', 'in', 'on', 'by', 'with', 'from', 'to', 'for', 'of', 'through']
    # for prep in prepositions:
    for prep in PREPOSITIONS:
        result = re.sub(fr'\b{prep}\b\s*([a-zA-Z])', fr'{prep} \1', result, flags=re.IGNORECASE)

    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s+([,.!?])', r'\1', result)

    return result.strip()


def extract_description_after_predicate(description: str) -> str:
    """从描述中提取谓语动词后的部分"""
    predicates = [
        r'\bis\b', r'\bwas\b', r'\bhas been\b', r'\bhave been\b',
        r'\bbecame\b', r'\bremains\b'
    ]

    pattern = '|'.join(predicates)
    match = re.search(pattern, description, re.IGNORECASE)
    if match:
        predicate_end = match.end()
        extracted = description[predicate_end:].strip()
        if len(extracted) < 3:
            return ""
        return extracted

    return ""


def clean_entity_description(description: str) -> str:
    """清理和标准化实体描述"""
    if not description:
        return ""

    description_stripped = description.strip()
    if re.match(r'^(a|an)\s+\w+', description_stripped, re.IGNORECASE):
        return description_stripped

    extracted = extract_description_after_predicate(description)
    if not extracted:
        return ""

    cleaned = extracted.strip(' .,')
    if len(cleaned) < 3:
        return ""

    return cleaned


def combine_entity_and_description(entity_name: str, description: str) -> str:
    """将实体名称和描述组合成自然的句子"""
    if not description:
        return entity_name

    combined = f"{entity_name} is {description}"
    combined = re.sub(r'\s+', ' ', combined)
    combined = re.sub(r'\.+', '.', combined)
    if not combined.endswith('.'):
        combined += '.'

    return combined


def process_description(raw_description: str, entity_name: str) -> str:
    """处理原始描述并返回格式化的句子"""
    cleaned_desc = clean_entity_description(raw_description)

    if not cleaned_desc:
        return ""

    return combine_entity_and_description(entity_name, cleaned_desc)


def verify_key_entities_consistency(original_entities, mutated_entities):
    """验证关键实体的一致性"""
    verification_results = {
        "missing_entities": [],
        "type_mismatched_entities": [],
        "consistent_entities": []
    }

    for entity_type, entities in original_entities.items():
        mutated_relevant_entities = mutated_entities.get(entity_type, [])
        mutated_texts = {entity[0] for entity in mutated_relevant_entities}

        for text, start, end in entities:
            if text not in mutated_texts:
                verification_results["missing_entities"].append((text, entity_type))
            else:
                for m_text, m_start, m_end in mutated_relevant_entities:
                    if m_text == text and entity_type not in mutated_entities:
                        verification_results["type_mismatched_entities"].append((text, entity_type))

            if text in mutated_texts and entity_type in mutated_entities:
                verification_results["consistent_entities"].append((text, entity_type))

    return verification_results
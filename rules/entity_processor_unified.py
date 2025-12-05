# entity_processor_unified.py

"""
统一的实体替换处理模块
"""

import traceback
from typing import Tuple, List, Dict
from collections import defaultdict
import requests
from flair.data import Sentence
from flair.models import SequenceTagger

from rules.constants import PREPOSITIONS
from rules.possessive_utils import select_pronoun_universal
from rules.entity_utils import (
    find_full_entity_span,
    adjust_sentence_structure,
    process_description,
    verify_key_entities_consistency
)
from rules.spacySimple import comprehensive_checker


class UnifiedEntityProcessor:
    """"""

    def __init__(self, ner_model: str = "conll3"):
        """
        Args:
            ner_model: NER模型类型 ("conll3", "ontonotes", "aws", "azure")
        """
        self.ner_model = ner_model.lower()
        self._init_model_specific_config()

    def _init_model_specific_config(self):
        """"""
        if self.ner_model == "conll3":
            from rules.descriptionProcessor import get_entity_description_conll3
            self.get_description_func = get_entity_description_conll3

        elif self.ner_model == "ontonotes":
            from rules.descriptionProcessor import get_entity_description_ontonotes
            self.get_description_func = get_entity_description_ontonotes

        elif self.ner_model == "aws":
            from rules.descriptionProcessor import get_entity_description_aws
            self.get_description_func = get_entity_description_aws

        elif self.ner_model == "azure":
            from rules.descriptionProcessor import get_entity_description_azure
            self.get_description_func = get_entity_description_azure
        else:
            raise ValueError(f"Unsupported NER model: {self.ner_model}")

    def get_entities(self, sentence_text: str, tagger: SequenceTagger) -> dict:
        """"""
        sentence = Sentence(sentence_text)
        tagger.predict(sentence)
        entities = defaultdict(list)

        for entity in sentence.get_spans('ner'):
            entities[entity.tag].append((entity.text, entity.start_position, entity.end_position))

        return entities

    def should_skip_entity_replacement(self, sentence: str, entity: tuple, entity_type: str) -> bool:
        """
        
        """
        return comprehensive_checker.should_skip_entity_replacement(
            sentence, entity, entity_type, self.ner_model
        )

    def replace_with_pronoun(self, sentence: str, entity: tuple, entity_type: str) -> Tuple[str, str]:
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
            context_before = sentence[:span.start].rstrip()

            print(f"\nDEBUG - Context after entity: '{context_after}'")
            print(f"DEBUG - Context before entity: '{context_before}'")

            # 
            pronoun = select_pronoun_universal(
                entity_type=entity_type,
                context_after=context_after,
                sentence=sentence,
                context_before=context_before,
                ner_model=self.ner_model
            )
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
                    # prepositions = {'at', 'in', 'on', 'by', 'with', 'from', 'to', 'for', 'of', 'through'}

                    print(f"DEBUG - Last word of prefix: '{last_word}'")
                    print(f"DEBUG - Is preposition: {last_word in PREPOSITIONS}")

                    # if last_word in prepositions:
                    if last_word in PREPOSITIONS:
                        prefix = ' '.join(words[:-1] + [last_word]) + ' '
                        print(f"DEBUG - Modified prefix after preposition: '{prefix}'")
                    elif not prefix[-1].isspace():
                        prefix += ' '
                        print(f"DEBUG - Added space to prefix: '{prefix}'")

            # 
            if suffix and not suffix[0] in " \n\t.,!?":
                suffix = ' ' + suffix
                print(f"DEBUG - Modified suffix with space: '{suffix}'")

            # 处理大小写
            print(f"Pronoun before capitalization: '{pronoun}'")
            needs_capitalization = not prefix or prefix.rstrip()[-1] in ".!?\n"
            print(f"Needs capitalization: {needs_capitalization}")
            if needs_capitalization:
                pronoun = pronoun.capitalize()
                print(f"Pronoun after capitalization: '{pronoun}'")

            # 构建最终句子
            new_sentence = prefix + pronoun + suffix
            print(f"\nDEBUG - Before final cleanup:")
            print(f"New sentence: '{new_sentence}'")

            # 最后的标点和空格清理
            new_sentence = adjust_sentence_structure(new_sentence)

            print(f"\nDEBUG - Final result:")
            print(f"Final sentence: '{new_sentence}'")

            return new_sentence, full_entity_text

        except Exception as e:
            print(f"Error in replace_with_pronoun: {str(e)}")
            traceback.print_exc()
            return sentence, entity_text

    def get_fallback_description(self, entity_type: str) -> str:
        """获取fallback描述"""
        from rules.descriptionProcessor import MultiNEREntityProcessor
        processor = MultiNEREntityProcessor()
        return processor._get_fallback_description(entity_type, self.ner_model)

    def mutate_and_verify(
            self,
            sentence_text: str,
            tagger: SequenceTagger,
            session=None
    ) -> Tuple[List[Dict], List[Dict]]:
        """执行变异和验证"""
        try:
            entities = self.get_entities(sentence_text, tagger)

            # 检查是否有任何实体
            has_entities = any(len(entity_list) > 0 for entity_list in entities.values())
            if not has_entities:
                return [], []

            # 确保session存在
            if session is None:
                session = requests.Session()
                session.proxies = {
                    "http": "http://127.0.0.1:7890",
                    "https": "http://127.0.0.1:7890",
                }
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json'
                })

            mutated_results = []
            suspicious_sentences = []

            for entity_type, entity_list in entities.items():
                if len(entity_list) == 0:
                    continue

                try:
                    first_entity = entity_list[0]

                    # 检查是否应该跳过这个实体的替换
                    if self.should_skip_entity_replacement(sentence_text, first_entity, entity_type):
                        continue

                    # 替换为代词
                    mutated_sentence, entity_name = self.replace_with_pronoun(
                        sentence_text, first_entity, entity_type
                    )

                    # 获取实体描述
                    raw_description = self.get_description_func(
                        entity_text=entity_name,
                        entity_type=entity_type,
                        session=session
                    )

                    entity_intro = process_description(raw_description, entity_name)

                    if not entity_intro:
                        entity_intro = f"{entity_name} is {self.get_fallback_description(entity_type)}."

                    combined_sentence = f"{entity_intro} {mutated_sentence}"
                    combined_entities = self.get_entities(combined_sentence, tagger)

                    result = {
                        "mutated_sentence": combined_sentence,
                        "entities": combined_entities,
                        "original_text": sentence_text
                    }
                    mutated_results.append(result)

                    # 验证实体一致性
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
"new test"
# entity_linking.py

"""
CoNLL3 NER
"""

from rules.entity_processor_unified import UnifiedEntityProcessor
from rules.spacySimple import comprehensive_checker

# 
_processor = UnifiedEntityProcessor("conll3")

# 
get_entities = _processor.get_entities
replace_with_pronoun = _processor.replace_with_pronoun
should_skip_entity_replacement = _processor.should_skip_entity_replacement


def mutate_and_verify_with_knowledge_graph(sentence_text: str, tagger, dbpedia_session=None):
    """"""
    return _processor.mutate_and_verify(sentence_text, tagger, dbpedia_session)


# 
from rules.entity_utils import (
    find_full_entity_span,
    adjust_sentence_structure,
    process_description,
    verify_key_entities_consistency,
    EntitySpan
)

__all__ = [
    'mutate_and_verify_with_knowledge_graph',
    'get_entities',
    'replace_with_pronoun',
    'should_skip_entity_replacement',
    'find_full_entity_span',
    'adjust_sentence_structure',
    'process_description',
    'verify_key_entities_consistency',
    'EntitySpan',
    'comprehensive_checker'
]
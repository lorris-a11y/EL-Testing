"new test"
# entity_linking4Aws.py

"""
AWS 
"""

from rules.entity_processor_unified import UnifiedEntityProcessor

# 
_processor = UnifiedEntityProcessor("aws")


def mutate_and_verify_with_knowledge_graph(sentence_text: str, tagger, dbpedia_session=None):
    """"""
    return _processor.mutate_and_verify(sentence_text, tagger, dbpedia_session)


# 
get_entities = _processor.get_entities
replace_with_pronoun = _processor.replace_with_pronoun
should_skip_entity_replacement = _processor.should_skip_entity_replacement
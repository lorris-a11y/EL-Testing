from flair.data import Sentence
from collections import defaultdict
from itertools import product

from rules.entity_swap_filters import should_filter_entity_for_swap


def extract_entities(sentence, tagger):
    """
    Extract entities and their tags from a sentence using the NER model.
    """
    sentence_obj = Sentence(sentence)
    tagger.predict(sentence_obj)
    entities = sentence_obj.get_spans("ner")
    entities_with_positions = [
        {"text": entity.text, "tag": entity.tag, "start": entity.start_position, "end": entity.end_position}
        for entity in entities
    ]
    return entities_with_positions

# Function to extract entities from text and return a set of entity texts
def extract_entity_texts(text, tagger):
    #
    entities = extract_entities(text, tagger)
    return set([entity["text"] for entity in entities])

#
def jaccard_similarity(set1, set2):
    """
    。
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if len(union) > 0 else 0




#
def swap_entities_across_sentences(sentence1, entities1, sentence2, entities2, ner_model="conll3"):
    """

    """
    swapped_sentences = []  # Store all swapped sentence pairs
    swapped_entities_list = []  # Track swapped entity pairs

    #
    filtered_entities1 = []
    filtered_entities2 = []

    print(f"\nDEBUG - Filtering entities for model: {ner_model}")

    #
    for entity in entities1:
        should_filter, reason = should_filter_entity_for_swap(entity, sentence1, ner_model)
        if should_filter:
            print(f"DEBUG - Filtered entity from sentence1: '{entity['text']}' - {reason}")
        else:
            filtered_entities1.append(entity)

    #
    for entity in entities2:
        should_filter, reason = should_filter_entity_for_swap(entity, sentence2, ner_model)
        if should_filter:
            print(f"DEBUG - Filtered entity from sentence2: '{entity['text']}' - {reason}")
        else:
            filtered_entities2.append(entity)

    print(
        f"DEBUG - After filtering: {len(filtered_entities1)} entities in s1, {len(filtered_entities2)} entities in s2")

    # Group filtered entities by their tags
    entities_by_tag = defaultdict(list)
    for e in filtered_entities1:
        entities_by_tag[e["tag"]].append(("s1", e))  # Mark from sentence 1
    for e in filtered_entities2:
        entities_by_tag[e["tag"]].append(("s2", e))  # Mark from sentence 2

    # Swap entities between the sentences for each tag
    for tag, entities in entities_by_tag.items():
        # Get all entities from each sentence
        s1_entities = [e for src, e in entities if src == "s1"]
        s2_entities = [e for src, e in entities if src == "s2"]

        # Perform pairwise swaps for all matching entities
        for s1_entity, s2_entity in product(s1_entities, s2_entities):
            # Create new versions of the sentences
            swapped_sentence1 = (
                    sentence1[:s1_entity["start"]]
                    + s2_entity["text"]
                    + sentence1[s1_entity["end"]:]
            )
            swapped_sentence2 = (
                    sentence2[:s2_entity["start"]]
                    + s1_entity["text"]
                    + sentence2[s2_entity["end"]:]
            )

            # Track the swapped entity pair
            swapped_entities = {
                "entity1": s1_entity["text"],
                "entity2": s2_entity["text"],
                "tag": tag,
            }

            # Store the result
            swapped_sentences.append((swapped_sentence1, swapped_sentence2))
            swapped_entities_list.append(swapped_entities)

    return swapped_sentences, swapped_entities_list






#
def mutate_and_verify_rule_two(original_sentence1, original_sentence2, tagger, ner_model=None):
    """

    """
    # Step 1:
    entities1 = extract_entities(original_sentence1, tagger)
    entities2 = extract_entities(original_sentence2, tagger)

    # Step 2:
    combined_sentence = f"{original_sentence1} {original_sentence2}"

    # Step 3:
    swapped_sentences, swapped_entities = swap_entities_across_sentences(
        original_sentence1, entities1, original_sentence2, entities2, ner_model
    )

    # Step 4:
    mutated_results = []
    suspicious_results = []


    original_entity_map1 = {entity["text"]: {"tag": entity["tag"], "position": (entity["start"], entity["end"])}
                            for entity in entities1}
    original_entity_map2 = {entity["text"]: {"tag": entity["tag"], "position": (entity["start"], entity["end"])}
                            for entity in entities2}

    #
    for idx, ((swapped_sentence1, swapped_sentence2), swap_info) in enumerate(zip(swapped_sentences, swapped_entities)):
        #
        mutated_combined_sentence = f"{swapped_sentence1} {swapped_sentence2}"

        #
        original_combined_entities = {}
        for entity in entities1:
            entity_type = entity["tag"]
            if entity_type not in original_combined_entities:
                original_combined_entities[entity_type] = []
            original_combined_entities[entity_type].append({
                "text": entity["text"],
                "start": entity["start"],
                "end": entity["end"]
            })

        for entity in entities2:
            entity_type = entity["tag"]
            # 调整第二个句子的位置
            adjusted_start = len(original_sentence1) + 1 + entity["start"]  # +1 for space
            adjusted_end = len(original_sentence1) + 1 + entity["end"]

            if entity_type not in original_combined_entities:
                original_combined_entities[entity_type] = []
            original_combined_entities[entity_type].append({
                "text": entity["text"],
                "start": adjusted_start,
                "end": adjusted_end
            })

        #
        mutated_entities1 = extract_entities(swapped_sentence1, tagger)
        mutated_entities2 = extract_entities(swapped_sentence2, tagger)

        #
        mutated_entity_map1 = {entity["text"]: {"tag": entity["tag"], "position": (entity["start"], entity["end"])}
                               for entity in mutated_entities1}
        mutated_entity_map2 = {entity["text"]: {"tag": entity["tag"], "position": (entity["start"], entity["end"])}
                               for entity in mutated_entities2}

        #
        mutated_combined_entities = {}
        for entity in mutated_entities1:
            entity_type = entity["tag"]
            if entity_type not in mutated_combined_entities:
                mutated_combined_entities[entity_type] = []
            mutated_combined_entities[entity_type].append({
                "text": entity["text"],
                "start": entity["start"],
                "end": entity["end"]
            })

        for entity in mutated_entities2:
            entity_type = entity["tag"]
            #
            adjusted_start = len(swapped_sentence1) + 1 + entity["start"]  # +1 for space
            adjusted_end = len(swapped_sentence1) + 1 + entity["end"]

            if entity_type not in mutated_combined_entities:
                mutated_combined_entities[entity_type] = []
            mutated_combined_entities[entity_type].append({
                "text": entity["text"],
                "start": adjusted_start,
                "end": adjusted_end
            })

        #
        mutated_results.append({
            "original_combined_sentence": combined_sentence,
            "mutated_combined_sentence": mutated_combined_sentence,
            "original_entities": original_combined_entities,
            "mutated_entities": mutated_combined_entities
        })

        #
        reasons = []

        # 1.
        for entity_text, info in original_entity_map1.items():
            #
            if swap_info["tag"] == info["tag"] and (
                    entity_text == swap_info["entity1"] or entity_text == swap_info["entity2"]):
                #
                if entity_text in mutated_entity_map2:
                    if mutated_entity_map2[entity_text]["tag"] != info["tag"]:
                        reasons.append(
                            f"Entity '{entity_text}' changed tag from '{info['tag']}' to '{mutated_entity_map2[entity_text]['tag']}' after swapping to sentence 2")
                else:
                    reasons.append(f"Entity '{entity_text}' missing from sentence 2 after swapping")
            else:
                #
                if entity_text in mutated_entity_map1:
                    if mutated_entity_map1[entity_text]["tag"] != info["tag"]:
                        reasons.append(
                            f"Entity '{entity_text}' changed tag from '{info['tag']}' to '{mutated_entity_map1[entity_text]['tag']}' in sentence 1 (not swapped)")
                else:
                    reasons.append(f"Entity '{entity_text}' missing from sentence 1 (not swapped)")

        #
        for entity_text, info in original_entity_map2.items():
            #
            if swap_info["tag"] == info["tag"] and (
                    entity_text == swap_info["entity1"] or entity_text == swap_info["entity2"]):
                #
                if entity_text in mutated_entity_map1:
                    if mutated_entity_map1[entity_text]["tag"] != info["tag"]:
                        reasons.append(
                            f"Entity '{entity_text}' changed tag from '{info['tag']}' to '{mutated_entity_map1[entity_text]['tag']}' after swapping to sentence 1")
                else:
                    reasons.append(f"Entity '{entity_text}' missing from sentence 1 after swapping")
            else:
                #
                if entity_text in mutated_entity_map2:
                    if mutated_entity_map2[entity_text]["tag"] != info["tag"]:
                        reasons.append(
                            f"Entity '{entity_text}' changed tag from '{info['tag']}' to '{mutated_entity_map2[entity_text]['tag']}' in sentence 2 (not swapped)")
                else:
                    reasons.append(f"Entity '{entity_text}' missing from sentence 2 (not swapped)")

        #
        for entity_text in mutated_entity_map1:
            if entity_text not in original_entity_map1 and entity_text not in original_entity_map2:
                reasons.append(
                    f"Unexpected new entity '{entity_text}' with tag '{mutated_entity_map1[entity_text]['tag']}' appeared in sentence 1")

        for entity_text in mutated_entity_map2:
            if entity_text not in original_entity_map1 and entity_text not in original_entity_map2:
                reasons.append(
                    f"Unexpected new entity '{entity_text}' with tag '{mutated_entity_map2[entity_text]['tag']}' appeared in sentence 2")

        #
        if reasons:
            suspicious_results.append({
                "original_sentence": combined_sentence,
                "mutated_sentence": mutated_combined_sentence,
                "reasons": reasons,
                "original_entities": original_combined_entities,
                "mutated_entities": mutated_combined_entities,
                "swapped_entity_type": swap_info["tag"]
            })

    return mutated_results, suspicious_results
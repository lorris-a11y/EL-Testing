"""
Enhanced entity reason parsing module - Extract entity information from complete suspicious results
Supports finding real tags from original_entities and mutated_entities
"""

import re
from typing import Dict, Optional, Any


def find_entity_tag_in_entities(entity_text: str, entities_dict: Dict[str, Any]) -> Optional[str]:
    """
    Find the tag of a specified entity in the entities dictionary

    Args:
        entity_text: The entity text to search for
        entities_dict: Entity dictionary in the format {tag: [{"text": ..., "start": ..., "end": ...}, ...]}

    Returns:
        The found entity tag, or None if not found
    """
    if not entities_dict:
        return None

    for tag, entity_list in entities_dict.items():
        if not isinstance(entity_list, list):
            continue

        for entity in entity_list:
            if isinstance(entity, dict) and entity.get("text") == entity_text:
                return tag

    return None


def extract_entity_info_from_reason_enhanced(reason: str, original_entities: Dict = None,
                                             mutated_entities: Dict = None) -> Optional[Dict]:
    """
    Extract entity information from suspicious reason descriptions, supporting real tag lookup in entities dictionaries

    Args:
        reason: Suspicious reason description
        original_entities: Original entity dictionary
        mutated_entities: Mutated entity dictionary

    Returns:
        Extracted entity information
    """
    if original_entities is None:
        original_entities = {}
    if mutated_entities is None:
        mutated_entities = {}

    # === Original matching patterns, but will attempt to find real tags from entities ===

    # Match "Entity 'X' changed tag from 'Y' to 'Z'"
    match1 = re.match(r"Entity '([^']+)' changed tag from '([^']+)' to '([^']+)'$", reason)
    if match1:
        entity_text = match1.group(1)
        original_tag = match1.group(2)
        mutated_tag = match1.group(3)

        # Verify if tags match those in entities, and use entities if they differ
        real_original_tag = find_entity_tag_in_entities(entity_text, original_entities)
        real_mutated_tag = find_entity_tag_in_entities(entity_text, mutated_entities)

        return {
            "entity_text": entity_text,
            "original_tag": real_original_tag or original_tag,
            "mutated_tag": real_mutated_tag or mutated_tag,
            "issue_type": "tag_change"
        }

    # Match "Entity 'X' of type 'Y' is missing"
    match2 = re.match(r"Entity '([^']+)' of type '([^']+)' is missing", reason)
    if match2:
        entity_text = match2.group(1)
        stated_tag = match2.group(2)

        # Find the real tag from original_entities
        real_original_tag = find_entity_tag_in_entities(entity_text, original_entities)

        return {
            "entity_text": entity_text,
            "original_tag": real_original_tag or stated_tag,
            "mutated_tag": "O",
            "issue_type": "missing"
        }

    # Match "Expected entity 'X' with tag 'Y' not found"
    match3 = re.match(r"Expected entity '([^']+)' with tag '([^']+)' not found", reason)
    if match3:
        entity_text = match3.group(1)
        stated_tag = match3.group(2)

        # Find the real tag from original_entities
        real_original_tag = find_entity_tag_in_entities(entity_text, original_entities)

        return {
            "entity_text": entity_text,
            "original_tag": real_original_tag or stated_tag,
            "mutated_tag": "O",
            "issue_type": "missing"
        }

    # Match "Entity 'X' disappeared" - This is a critical fix point
    match4 = re.match(r"Entity '([^']+)' disappeared", reason)
    if match4:
        entity_text = match4.group(1)

        # Find the real tag from original_entities
        real_original_tag = find_entity_tag_in_entities(entity_text, original_entities)

        return {
            "entity_text": entity_text,
            "original_tag": real_original_tag or "UNKNOWN",  # Set to UNKNOWN only if not found
            "mutated_tag": "O",
            "issue_type": "missing"
        }

    # === MR2-specific matching patterns ===

    # Match "Entity 'X' changed tag from 'Y' to 'Z' after swapping to sentence N"
    match5 = re.match(r"Entity '([^']+)' changed tag from '([^']+)' to '([^']+)' after swapping to sentence \d+",
                      reason)
    if match5:
        entity_text = match5.group(1)
        original_tag = match5.group(2)
        mutated_tag = match5.group(3)

        # Verify tags
        real_original_tag = find_entity_tag_in_entities(entity_text, original_entities)
        real_mutated_tag = find_entity_tag_in_entities(entity_text, mutated_entities)

        return {
            "entity_text": entity_text,
            "original_tag": real_original_tag or original_tag,
            "mutated_tag": real_mutated_tag or mutated_tag,
            "issue_type": "tag_change"
        }

    # Match "Entity 'X' changed tag from 'Y' to 'Z' in sentence N (not swapped)"
    match6 = re.match(r"Entity '([^']+)' changed tag from '([^']+)' to '([^']+)' in sentence \d+ \(not swapped\)",
                      reason)
    if match6:
        entity_text = match6.group(1)
        original_tag = match6.group(2)
        mutated_tag = match6.group(3)

        # Verify tags
        real_original_tag = find_entity_tag_in_entities(entity_text, original_entities)
        real_mutated_tag = find_entity_tag_in_entities(entity_text, mutated_entities)

        return {
            "entity_text": entity_text,
            "original_tag": real_original_tag or original_tag,
            "mutated_tag": real_mutated_tag or mutated_tag,
            "issue_type": "tag_change"
        }

    # Match "Entity 'X' missing from sentence N after swapping"
    match7 = re.match(r"Entity '([^']+)' missing from sentence \d+ after swapping", reason)
    if match7:
        entity_text = match7.group(1)

        # Find the real tag from original_entities
        real_original_tag = find_entity_tag_in_entities(entity_text, original_entities)

        return {
            "entity_text": entity_text,
            "original_tag": real_original_tag or "UNKNOWN",
            "mutated_tag": "O",
            "issue_type": "missing"
        }

    # Match "Entity 'X' missing from sentence N (not swapped)"
    match8 = re.match(r"Entity '([^']+)' missing from sentence \d+ \(not swapped\)", reason)
    if match8:
        entity_text = match8.group(1)

        # Find the real tag from original_entities
        real_original_tag = find_entity_tag_in_entities(entity_text, original_entities)

        return {
            "entity_text": entity_text,
            "original_tag": real_original_tag or "UNKNOWN",
            "mutated_tag": "O",
            "issue_type": "missing"
        }

    # Match "Unexpected new entity 'X' with tag 'Y' appeared in sentence N"
    match9 = re.match(r"Unexpected new entity '([^']+)' with tag '([^']+)' appeared in sentence \d+", reason)
    if match9:
        entity_text = match9.group(1)
        stated_tag = match9.group(2)

        # Verify the tag in mutated_entities
        real_mutated_tag = find_entity_tag_in_entities(entity_text, mutated_entities)

        return {
            "entity_text": entity_text,
            "original_tag": "O",  # Did not exist originally
            "mutated_tag": real_mutated_tag or stated_tag,
            "issue_type": "new_entity"
        }

    return None


# For backward compatibility, keep the original function
def extract_entity_info_from_reason(reason: str) -> Optional[Dict]:
    """Backward-compatible function, calls the enhanced version without passing entities information"""
    return extract_entity_info_from_reason_enhanced(reason, None, None)
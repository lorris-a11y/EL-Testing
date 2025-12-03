def match_case_for_candidate(candidate_text: str, original_entity_text: str) -> str:
    """
    Match the case of the first letter of the candidate text to the first letter of the original entity text

    Args:
        candidate_text: The candidate entity text to modify
        original_entity_text: The original entity text to match case with

    Returns:
        The candidate text with first letter case matching the original entity
    """
    if not candidate_text or not original_entity_text:
        return candidate_text

    # Check if the original entity starts with an uppercase letter
    if original_entity_text[0].isupper():
        return candidate_text[0].upper() + candidate_text[1:] if len(candidate_text) > 1 else candidate_text[0].upper()
    else:
        return candidate_text[0].lower() + candidate_text[1:] if len(candidate_text) > 1 else candidate_text[0].lower()
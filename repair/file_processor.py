
import json
from typing import List, Dict
from entity_repair import repair_entity, repair_entity_mr1
from reason_parser import extract_entity_info_from_reason_enhanced


def process_suspicious_file(suspicious_file: str, tagger, rule_type="default", model_type="flair") -> List[Dict]:
    """
    Process suspicious file and repair entities
    """
    with open(suspicious_file, 'r', encoding='utf-8') as f:
        suspicious_results = json.load(f)

    fixed_results = []

    for suspicious in suspicious_results:
        original_sentence = suspicious.get("original_sentence", "")
        mutated_sentence = suspicious.get("mutated_sentence", "")
        reasons = suspicious.get("reasons", [])

        original_entities = suspicious.get("original_entities", {})
        mutated_entities = suspicious.get("mutated_entities", {})

        sentence_fixes = []

        for reason in reasons:
            # Parse entity information from the reason
            entity_info = extract_entity_info_from_reason_enhanced(
                reason, original_entities, mutated_entities
            )

            if not entity_info:
                print(f"Warning: Unable to parse reason '{reason}'")
                continue

            print(f"Parsed entity information: {entity_info}")

            # Select the appropriate repair function based on the rule type
            if rule_type == "MR1":
                corrected_tag, confidence = repair_entity_mr1(
                    original_sentence,
                    mutated_sentence,              
                    entity_info["entity_text"],
                    entity_info["entity_text"],    
                    entity_info["original_tag"],
                    entity_info["mutated_tag"],
                    tagger,
                    model_type
                )
            else:
                corrected_tag, confidence = repair_entity(
                    original_sentence,
                    mutated_sentence,   
                    entity_info["entity_text"],
                    entity_info["entity_text"], 
                    entity_info["original_tag"],
                    entity_info["mutated_tag"],
                    tagger,
                    model_type
                )

            # Display the final result
            print(f"Entity '{entity_info['entity_text']}' repaired: {entity_info['original_tag']} → {corrected_tag} (Confidence: {confidence:.3f})")
            print("-" * 80)  # Separator

            sentence_fixes.append({
                "entity": entity_info["entity_text"],
                "original_tag": entity_info["original_tag"],
                "mutated_tag": entity_info["mutated_tag"],
                "corrected_tag": corrected_tag,
                "confidence": confidence,
                "issue_type": entity_info.get("issue_type", "unknown")
            })

        fixed_results.append({
            "original_sentence": original_sentence,
            "mutated_sentence": mutated_sentence,
            "repairs": sentence_fixes
        })

    return fixed_results


def run_repair_from_file(suspicious_file: str, output_file: str, model_path: str, rule_type="default", model_type="flair"):

    from flair.models import SequenceTagger
    import json
    import os

    # Load the NER model
    print(f"Loading NER model from {model_path}...")
    tagger = SequenceTagger.load(model_path)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Process the suspicious file
    print(f"Processing suspicious file: {suspicious_file}")
    fixed_results = process_suspicious_file(suspicious_file, tagger, rule_type, model_type)  # Add model_type parameter

    # Save the repair results
    print(f"Saving fixed results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_results, f, ensure_ascii=False, indent=4)

    # Statistics for repair results
    total_repairs = sum(len(result["repairs"]) for result in fixed_results)
    original_restored = sum(
        1 for result in fixed_results
        for repair in result["repairs"]
        if repair["corrected_tag"] == repair["original_tag"]
    )
    mutated_kept = sum(
        1 for result in fixed_results
        for repair in result["repairs"]
        if repair["corrected_tag"] == repair["mutated_tag"] and repair["mutated_tag"] != repair["original_tag"]
    )
    new_tag = total_repairs - original_restored - mutated_kept

    print("\nRepair statistics:")
    print(f"Total sentences processed: {len(fixed_results)}")
    print(f"Total entities attempted to repair: {total_repairs}")

    if total_repairs > 0:
        original_percent = (original_restored / total_repairs * 100)
        print(f"Restored original tags: {original_restored} ({original_percent:.1f}%)")

        mutated_percent = (mutated_kept / total_repairs * 100)
        print(f"Kept mutated tags: {mutated_kept} ({mutated_percent:.1f}%)")

        new_tag_percent = (new_tag / total_repairs * 100)
        print(f"Assigned new tags: {new_tag} ({new_tag_percent:.1f}%)")

    # Statistics for issue types
    issue_types = {}
    for result in fixed_results:
        for repair in result["repairs"]:
            issue_type = repair.get("issue_type", "unknown")
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

    print(f"\nIssue type statistics:")
    for issue_type, count in issue_types.items():
        print(f"  {issue_type}: {count}")


def run_repair_from_file_with_tagger(suspicious_file: str, output_file: str, tagger, rule_type="default",
                                     model_type="cloud"):

    import json
    import os

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Process the suspicious file
    print(f"Processing suspicious file: {suspicious_file}")
    fixed_results = process_suspicious_file(suspicious_file, tagger, rule_type, model_type)  # Add model_type parameter

    # Save the repair results
    print(f"Saving fixed results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_results, f, ensure_ascii=False, indent=4)

    # Statistics for repair results
    total_repairs = sum(len(result["repairs"]) for result in fixed_results)
    original_restored = sum(
        1 for result in fixed_results
        for repair in result["repairs"]
        if repair["corrected_tag"] == repair["original_tag"]
    )
    mutated_kept = sum(
        1 for result in fixed_results
        for repair in result["repairs"]
        if repair["corrected_tag"] == repair["mutated_tag"] and repair["mutated_tag"] != repair["original_tag"]
    )
    new_tag = total_repairs - original_restored - mutated_kept

    print("\nRepair statistics:")
    print(f"Total sentences processed: {len(fixed_results)}")
    print(f"Total entities attempted to repair: {total_repairs}")

    if total_repairs > 0:
        original_percent = (original_restored / total_repairs * 100)
        print(f"Restored original tags: {original_restored} ({original_percent:.1f}%)")

        mutated_percent = (mutated_kept / total_repairs * 100)
        print(f"Kept mutated tags: {mutated_kept} ({mutated_percent:.1f}%)")

        new_tag_percent = (new_tag / total_repairs * 100)
        print(f"Assigned new tags: {new_tag} ({new_tag_percent:.1f}%)")

    # Statistics for issue types
    issue_types = {}
    for result in fixed_results:
        for repair in result["repairs"]:
            issue_type = repair.get("issue_type", "unknown")
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

    print(f"\nIssue type statistics:")
    for issue_type, count in issue_types.items():
        print(f"  {issue_type}: {count}")
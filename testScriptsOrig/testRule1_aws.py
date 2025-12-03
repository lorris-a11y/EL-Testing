
import json
import re
import os
import time
import logging
import datetime
import traceback
from collections import defaultdict

# Import AWS tools
from AwsUtils.aws_ner_utils import setup_aws_mock, initialize_aws_tagger, AWSSentence

# Generate log filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"TR-AWS/testRule1-{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Set up AWS mock to simulate Flair interface
setup_aws_mock()


# Create get_entities function for AWS
def get_entities(sentence_text, tagger):
    """Get entities from sentence text using AWS tagger"""
    sentence = AWSSentence(sentence_text)
    tagger.predict(sentence)

    entities = defaultdict(list)
    for entity in sentence.get_spans('ner'):
        entities[entity.tag].append({
            'text': entity.text,
            'start': entity.start_position,
            'end': entity.end_position
        })

    return entities


# Define function to find coordinated entities, logic consistent with provided code
def find_coordinated_entities(sentence_text, tagger):
    """Find coordinated entities following the original mutationRuleOne approach, adapted for AWS NER"""
    # Get all entities
    entities = get_entities(sentence_text, tagger)
    coordinated_groups = []

    logger.debug("Identified entities:")
    for entity_type, entity_list in entities.items():
        entity_info = [f"{e['text']} ({e['start']}, {e['end']})" for e in entity_list]
        logger.debug(f"{entity_type}: {entity_info}")

    # Check if the entire sentence contains "between"
    has_between = "between" in sentence_text.lower()
    has_as_well_as = "as well as" in sentence_text.lower()

    # Check for parallel relationships for each entity type
    for entity_type, entity_list in entities.items():
        if len(entity_list) < 2:
            continue

        # Sort entities by position
        ents_sorted = sorted(entity_list, key=lambda x: x['start'])

        i = 0
        while i < len(ents_sorted) - 1:
            current_ent = ents_sorted[i]
            next_ent = ents_sorted[i + 1]

            # Get text between entities
            text_between = sentence_text[current_ent['end']:next_ent['start']]

            # Check if contains "as well as"
            if "as well as" in text_between.lower():
                logger.debug(f"Skipping 'as well as' structure: {current_ent['text']} as well as {next_ent['text']}")
                i += 2  # Skip this pair of entities
                continue

            # Check if "as well as" appears before current entity
            text_before = sentence_text[max(0, current_ent['start'] - 30):current_ent['start']]
            if has_as_well_as and "as well as" in text_before.lower():
                text_after_as_well_as = text_before.split("as well as")[-1].strip()
                if len(text_after_as_well_as) < 20:  # Assume entity follows "as well as"
                    logger.debug(f"Skipping 'X as well as' structure: X as well as {current_ent['text']}")
                    i += 1
                    continue

            # Check if currency range expression
            if has_between and entity_type in ["MONEY", "QUANTITY", "NUMBER"]:
                text_before = sentence_text[max(0, current_ent['start'] - 15):current_ent['start']]
                if "between" in text_before.lower():
                    logger.debug(f"Skipping range expression: between {current_ent['text']} and {next_ent['text']}")
                    i += 2  # Skip this pair of entities
                    continue

            # Check if "from X to Y" structure
            if entity_type in ["PRODUCT", "ORGANIZATION", "LOCATION"]:
                text_before = sentence_text[max(0, current_ent['start'] - 20):current_ent['start']]
                if "from" in text_before.lower() and "to" in text_between.lower():
                    logger.debug(f"Skipping from-to structure: from {current_ent['text']} to {next_ent['text']}")
                    i += 2  # Skip this pair of entities
                    continue

            # Check if in special quote structure
            in_special_quote = False
            # Check if in question (like "what in the bananas")
            if re.search(r'\b(?:what|which|where|who|when|how)\b', text_between, re.IGNORECASE):
                # Only check if quotes appear before or after
                text_chunk = sentence_text[
                             max(0, current_ent['start'] - 30):min(len(sentence_text), next_ent['end'] + 30)]
                if ('"' in text_chunk or "'" in text_chunk or "?" in text_chunk):
                    in_special_quote = True

            if in_special_quote:
                logger.debug(f"Skipping special structure in quotes: {current_ent['text']} ... {next_ent['text']}")
                i += 1
                continue

            # Use strict parallel pattern matching, consistent with original code
            is_parallel = bool(
                re.search(r'^\s*,\s*(and\s+)?$|^\s*,?\s+and\s+$|^\s*,\s*$', text_between, re.IGNORECASE) and
                not re.search(r'\b(?:which|that|who|when|where|because)\b', text_between, re.IGNORECASE) and
                not re.search(r'\b(?:is|was|are|were|has|have|had)\b', text_between, re.IGNORECASE)
            )

            if is_parallel:
                logger.debug(f"Found parallel structure: {current_ent['text']} and {next_ent['text']}")
                current_group = [current_ent]
                j = i + 1
                while j < len(ents_sorted):
                    text_between = sentence_text[ents_sorted[j - 1]['end']:ents_sorted[j]['start']]
                    if re.search(r'^\s*,\s*(and\s+)?$|^\s*,?\s+and\s+$|^\s*,\s*$', text_between, re.IGNORECASE):
                        current_group.append(ents_sorted[j])
                        j += 1
                    else:
                        break

                if len(current_group) > 1:
                    coordinated_groups.append({
                        'type': entity_type,
                        'entities': current_group
                    })
                i = j
            else:
                i += 1

    logger.debug(f"Found {len(coordinated_groups)} coordinated entity groups")
    for i, group in enumerate(coordinated_groups):
        logger.debug(f"Group {i + 1} - Type: {group['type']}")
        for entity in group['entities']:
            logger.debug(f"  Entity: '{entity['text']}' ({entity['start']}, {entity['end']})")

    return coordinated_groups


# Save original function and replace with our version
import rules.mutationRuleOne

original_function = rules.mutationRuleOne.find_coordinated_entities
rules.mutationRuleOne.find_coordinated_entities = find_coordinated_entities

# Import mutation function
from rules.mutationRuleOne import mutate_and_verify


def process_mutations_rule_one_aws():
    """Process mutations with rule one and AWS NER"""
    # Record start time
    start_time = time.time()
    logger.info(f"Starting process_mutations_rule_one_aws (Log file: {log_filename})")

    # Set up AWS mock to simulate Flair interface
    setup_aws_mock()

    # Initialize AWS tagger
    logger.info("Initializing AWS tagger")
    tagger = initialize_aws_tagger()
    logger.info("AWS tagger initialized successfully")

    # Read JSON file
    logger.info("Reading data from JSON file")
    try:
        with open('your_test_file_here', 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON data with {len(data)} records")
    except Exception as e:
        logger.error(f"Error reading JSON file: {e}")
        return

    all_mutated_results = []
    all_suspicious_results = []

    # Create output directory (if it doesn't exist)
    os.makedirs('../TR-AWS', exist_ok=True)

    # Use fixed result filenames
    mutated_results_filename = 'your_mutated_file_path'
    suspicious_results_filename = 'your_suspicious_file_path'

    # Process each entry
    logger.info("Starting mutation processing")
    for item_idx, item in enumerate(data):
        if item_idx % 10 == 0 and item_idx > 0:  # Log progress every 10 records
            logger.info(f"Processed {item_idx}/{len(data)} records")

        original_text = item.get('sentence', '')
        # original_text = item.get('original', '')


        if not original_text:
            logger.warning(f"Empty text found at index {item_idx}")
            continue

        # Apply rule 1
        try:
            # Record single processing start time
            mutation_start_time = time.time()

            # Call original mutation function (using rewritten find_coordinated_entities)
            mutated_results_1, suspicious_results_1 = mutate_and_verify(original_text, tagger)

            # Calculate single processing time
            mutation_time = time.time() - mutation_start_time

            if mutation_time > 5:  # Log if single processing time exceeds 5 seconds
                logger.warning(f"Long processing time ({mutation_time:.2f}s) for text: {original_text[:50]}...")

            # Process mutation results
            processed_mutated_results = []
            for result in mutated_results_1:
                processed_result = {
                    "original_sentence": original_text,
                    "mutated_sentence": result["mutated_sentence"],
                    "entities": {
                        tag: [{"text": e["text"], "start": e["start"], "end": e["end"]}
                              for e in entities]
                        for tag, entities in result["entities"].items()
                    }
                }
                processed_mutated_results.append(processed_result)

            # Process suspicious results
            processed_suspicious_results = []
            for result in suspicious_results_1:
                processed_result = {
                    "original_sentence": original_text,
                    "mutated_sentence": result["mutated_sentence"],
                    "reasons": result.get("reasons", []),  # Include reasons (if any)
                    "original_entities": {
                        tag: [{"text": e["text"], "start": e["start"], "end": e["end"]}
                              for e in entities]
                        for tag, entities in result["original_entities"].items()
                    },
                    "mutated_entities": {
                        tag: [{"text": e["text"], "start": e["start"], "end": e["end"]}
                              for e in entities]
                        for tag, entities in result["mutated_entities"].items()
                    }
                }
                processed_suspicious_results.append(processed_result)

            all_mutated_results.extend(processed_mutated_results)
            all_suspicious_results.extend(processed_suspicious_results)

            logger.debug(
                f"Generated {len(processed_mutated_results)} mutation results and {len(processed_suspicious_results)} suspicious results")

        except Exception as e:
            logger.error(f"Error processing sentence: {original_text[:100]}...")
            logger.error(f"Error details: {str(e)}")
            logger.error(traceback.format_exc())

    logger.info(f"Mutation process completed. Results from {len(data)} records processed.")

    # Write all mutated sentences and recognition results to file
    logger.info(f"Writing mutation results to file: {mutated_results_filename}")
    try:
        with open(mutated_results_filename, 'w', encoding='utf-8') as f:
            json.dump(all_mutated_results, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved {len(all_mutated_results)} mutation results")
    except Exception as e:
        logger.error(f"Error saving mutation results: {e}")

    # Write all suspicious sentences to file
    logger.info(f"Writing suspicious sentences to file: {suspicious_results_filename}")
    try:
        with open(suspicious_results_filename, 'w', encoding='utf-8') as f:
            json.dump(all_suspicious_results, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved {len(all_suspicious_results)} suspicious results")
    except Exception as e:
        logger.error(f"Error saving suspicious results: {e}")

    # Calculate and print total execution time
    execution_time = time.time() - start_time

    # Print statistics to console
    num_suspicious_sentences = len(all_suspicious_results)
    logger.info(f"Total suspicious sentences: {num_suspicious_sentences}")
    logger.info(f"Mutation results saved to '{mutated_results_filename}'")
    logger.info(f"Suspicious sentences saved to '{suspicious_results_filename}'")
    logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time / 60:.2f} minutes)")


if __name__ == "__main__":
    try:
        logger.info(f"=== Starting testRule1_aws.py (Log: {log_filename}) ===")
        process_mutations_rule_one_aws()
        logger.info("=== Completed testRule1_aws.py ===")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
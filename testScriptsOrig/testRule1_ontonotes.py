import os
import json
import time
import logging
import datetime
from flair.models import SequenceTagger
from rules.mutationRuleOne import mutate_and_verify

# ==================== Configuration Parameters ====================
MODEL_PATH = ''
INPUT_FILE = ''
OUTPUT_DIR = ''
LOG_DIR = 'TR-Mode2'
# ==================================================================

# Generate log filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'{LOG_DIR}/testRule1-{timestamp}.log'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def process_mutations_rule_one():
    start_time = time.time()
    logger.info(f"Starting process_mutations_rule_one (Log file: {log_filename})")

    # Load the NER model
    logger.info("Loading NER model")
    logger.info(f"Model path: {MODEL_PATH}")

    tagger = SequenceTagger.load(MODEL_PATH)
    logger.info("NER model loaded successfully")

    # Read text from JSON file
    logger.info("Reading data from JSON file")
    logger.info(f"Input file: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON data with {len(data)} items")
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return

    all_mutated_results = []
    all_suspicious_results = []

    # Iterate over each item in the list
    logger.info("Starting mutation process")
    for i, item in enumerate(data):
        if i % 10 == 0 and i > 0:
            logger.info(f"Processed {i}/{len(data)} items")

        original_text = item.get('original', '')

        # Apply Rule 1
        mutation_start_time = time.time()
        mutated_results_1, suspicious_results_1 = mutate_and_verify(original_text, tagger)
        mutation_time = time.time() - mutation_start_time

        if mutation_time > 5:
            logger.warning(f"Long mutation time ({mutation_time:.2f}s) for text: {original_text[:50]}...")

        # Process mutated results
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
                "reasons": result.get("reasons", []),
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

    logger.info(f"Mutation process completed. Processing results for {len(data)} items.")

    # Generate output file paths
    mutated_results_filename = f'{OUTPUT_DIR}/mutated_results1_test.json'
    suspicious_results_filename = f'{OUTPUT_DIR}/suspicious_sentences1_test.json'

    # Write all mutated sentences and recognition results to file
    logger.info(f"Writing mutated results to file: {mutated_results_filename}")
    try:
        with open(mutated_results_filename, 'w', encoding='utf-8') as f:
            json.dump(all_mutated_results, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved {len(all_mutated_results)} mutated results")
    except Exception as e:
        logger.error(f"Error saving mutated results: {e}")

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
        logger.info(f"=== Starting testRule1_ontonotes.py (Log: {log_filename}) ===")
        logger.info(f"Configuration:")
        logger.info(f"  MODEL_PATH: {MODEL_PATH}")
        logger.info(f"  INPUT_FILE: {INPUT_FILE}")
        logger.info(f"  OUTPUT_DIR: {OUTPUT_DIR}")
        logger.info(f"  LOG_DIR: {LOG_DIR}")
        process_mutations_rule_one()
        logger.info("=== Completed testRule1_ontonotes.py ===")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
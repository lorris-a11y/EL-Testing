import json
import random
import os
import time
import logging
import datetime
from itertools import combinations
import traceback

# Import Azure utilities
from AzureUtils.azure_ner_utils import setup_azure_mock, initialize_azure_tagger, ensure_output_dir

# Set up a filter for Azure API request info before setting up the Azure mock
class AzureAPIFilter(logging.Filter):
    def filter(self, record):
        # Filter out specific Azure API information logs
        if 'Response headers' in record.getMessage():
            return False
        if 'Response status' in record.getMessage():
            return False
        if 'Content-Length' in record.getMessage():
            return False
        if 'apim-request-id' in record.getMessage():
            return False
        if 'x-envoy-upstream-service-time' in record.getMessage():
            return False
        if 'Set-Cookie' in record.getMessage():
            return False
        # Add more filtering conditions as needed
        return True

# Configure base logger
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'TR-Azure/testRule2-{timestamp}.log'

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create file handler
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# Add filter
file_handler.addFilter(AzureAPIFilter())

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# Add filter
console_handler.addFilter(AzureAPIFilter())

# Add handlers to root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Get logger for this program
logger = logging.getLogger(__name__)

# Now set up Azure mock
setup_azure_mock()

# Import necessary functions
from rules.mutationRuleTwo import extract_entity_texts, jaccard_similarity, mutate_and_verify_rule_two


def process_mutations_rule_two_azure():
    """Main function to process mutations using Rule Two with Azure NER."""
    # Record start time
    start_time = time.time()
    logger.info(f"Starting process_mutations_rule_two_azure (Log file: {log_filename})")

    # Initialize Azure tagger
    logger.info("Initializing Azure tagger")
    tagger = initialize_azure_tagger()
    logger.info("Azure tagger initialized successfully")

    # Read input data
    logger.info("Reading data from JSON file")
    try:
        with open('your_test_file_here', 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON data with {len(data)} items")
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return

    all_mutated_results = []
    all_suspicious_results = []
    jaccard_threshold = 0.5  # Set threshold
    num_random_pairs = 5000  # Set number of random combinations to test

    # Get all possible sentence pairs
    logger.info("Generating sentence pairs")
    all_sentence_pairs = list(combinations(data, 2))
    logger.info(f"Total possible sentence pairs: {len(all_sentence_pairs)}")

    # Randomly select specific number of sentence pairs
    logger.info(f"Randomly selecting {num_random_pairs} sentence pairs")
    random_pairs = random.sample(all_sentence_pairs, min(num_random_pairs, len(all_sentence_pairs)))

    # Ensure output directory exists
    ensure_output_dir('TR-Azure')

    # Create file to store Jaccard similarity results
    jaccard_file_path = 'testAzure/jaccard_results.txt'
    logger.info(f"Creating Jaccard similarity results file: {jaccard_file_path}")

    try:
        with open(jaccard_file_path, 'w', encoding='utf-8') as jaccard_file:
            # Process each random pair
            logger.info("Starting to process sentence pairs")
            for i, (sentence1, sentence2) in enumerate(random_pairs):
                if i % 100 == 0 and i > 0:  # Log progress every 100 records
                    logger.info(f"Processed {i}/{len(random_pairs)} sentence pairs")

                original_text1 = sentence1.get('original', '')
                original_text2 = sentence2.get('original', '')

                # Step 1: Calculate Jaccard similarity
                entities_text1 = extract_entity_texts(original_text1, tagger)
                entities_text2 = extract_entity_texts(original_text2, tagger)

                jaccard_score = jaccard_similarity(entities_text1, entities_text2)

                # Write Jaccard similarity to file
                jaccard_file.write(f"Jaccard Similarity: {jaccard_score:.4f}\n")
                jaccard_file.write("\n" + "-" * 80 + "\n\n")

                # Step 2: If Jaccard similarity is above threshold, skip this combination
                if jaccard_score > jaccard_threshold:
                    logger.debug(f"Skipping combination: Jaccard similarity {jaccard_score:.4f} > threshold")
                    continue

                # Step 3: If Jaccard similarity is below threshold, continue with mutation testing
                try:
                    mutation_start_time = time.time()
                    mutated_results_2, suspicious_results_2 = mutate_and_verify_rule_two(original_text1, original_text2, tagger, "azure")
                    mutation_time = time.time() - mutation_start_time

                    # Log suspicious results
                    if suspicious_results_2:
                        logger.info(f"Found {len(suspicious_results_2)} suspicious results for sentence pair")
                        if len(suspicious_results_2) > 0:
                            sample = suspicious_results_2[0]
                            if "reasons" in sample and sample["reasons"]:
                                logger.info(f"Sample suspicious reason: {sample['reasons'][0]}")
                            elif "original_sentence" in sample:
                                logger.info(f"Sample suspicious sentence: {sample['original_sentence'][:50]}...")

                    # Add mutated results directly (already in correct format)
                    all_mutated_results.extend(mutated_results_2)

                    # Add suspicious results directly
                    all_suspicious_results.extend(suspicious_results_2)
                except Exception as e:
                    logger.error(f"Error processing sentence pair: {str(e)}")
                    logger.error(f"Sentence 1: {original_text1}")
                    logger.error(f"Sentence 2: {original_text2}")
                    logger.error(traceback.format_exc())

        logger.info(f"Finished processing all sentence pairs. Total suspicious results: {len(all_suspicious_results)}")

    except Exception as e:
        logger.error(f"Error processing sentence pairs: {e}", exc_info=True)
        return

    # Save result files
    mutated_results_filename = 'your_mutated_file_path'
    suspicious_results_filename = 'your_suspicious_file_path'

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

    # Print statistics
    num_suspicious_sentences = len(all_suspicious_results)
    num_suspicious_entities = sum(len(susp.get("reasons", [])) for susp in all_suspicious_results)
    logger.info(f"Total suspicious sentences: {num_suspicious_sentences}")
    logger.info(f"Total suspicious entities: {num_suspicious_entities}")
    logger.info(f"Mutation results saved to '{mutated_results_filename}'")
    logger.info(f"Suspicious sentences saved to '{suspicious_results_filename}'")
    logger.info(f"Jaccard similarity results saved to '{jaccard_file_path}'")
    logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time / 60:.2f} minutes)")


if __name__ == "__main__":
    try:
        logger.info(f"=== Starting testRule2Azure.py (Log: {log_filename}) ===")
        process_mutations_rule_two_azure()
        logger.info("=== Completed testRule2Azure.py ===")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
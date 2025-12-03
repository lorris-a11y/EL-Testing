import json
import random
import time
import logging
import datetime
import argparse
import sys
from flair.models import SequenceTagger
from rules.mutationRuleTwo import mutate_and_verify_rule_two, extract_entity_texts, \
    jaccard_similarity

from collections import defaultdict
from itertools import combinations

# ==================== Configuration Parameters ====================
DEFAULT_MODEL_PATH = ''
DEFAULT_OUTPUT_DIR = ''
DEFAULT_INPUT_FILE = ''
DEFAULT_LOG_DIR = 'TR-Mode2'

ENABLE_ENTITY_FILTER = True
NER_MODEL_NAME = "ontonotes"

JACCARD_THRESHOLD = 0.5
NUM_RANDOM_PAIRS = 5000
# ==================================================================

def process_mutations_rule_two(model_path=None, output_dir=None, input_file=None):
    # Use default values if not provided
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    if input_file is None:
        input_file = DEFAULT_INPUT_FILE

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'{output_dir}/testRule2-{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ],
        force=True
    )

    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info(f"Starting process_mutations_rule_two (Log file: {log_filename})")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Entity filter enabled: {ENABLE_ENTITY_FILTER}")
    if ENABLE_ENTITY_FILTER:
        logger.info(f"Using NER model for filtering: {NER_MODEL_NAME}")

    # Load the NER model
    logger.info("Loading NER model")
    tagger = SequenceTagger.load(model_path)
    logger.info("NER model loaded successfully")

    logger.info("Reading data from JSON file")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON data with {len(data)} items")
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return

    all_mutated_results = []
    all_suspicious_results = []

    logger.info("Generating sentence pairs")
    all_sentence_pairs = list(combinations(data, 2))
    logger.info(f"Total possible sentence pairs: {len(all_sentence_pairs)}")

    logger.info(f"Randomly selecting {NUM_RANDOM_PAIRS} sentence pairs")
    random_pairs = random.sample(all_sentence_pairs, min(NUM_RANDOM_PAIRS, len(all_sentence_pairs)))

    # Ensure output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)

    jaccard_file_path = f'{output_dir}/jaccard_results.txt'
    logger.info(f"Creating Jaccard similarity results file: {jaccard_file_path}")

    total_pairs_processed = 0
    pairs_filtered_by_jaccard = 0
    pairs_with_mutations = 0
    pairs_with_suspicious = 0

    try:
        with open(jaccard_file_path, 'w', encoding='utf-8') as jaccard_file:

            logger.info("Starting to process sentence pairs")
            for i, (sentence1, sentence2) in enumerate(random_pairs):
                if i % 100 == 0 and i > 0:
                    logger.info(f"Processed {i}/{len(random_pairs)} sentence pairs")
                    logger.info(f"  - Filtered by Jaccard: {pairs_filtered_by_jaccard}")
                    logger.info(f"  - Pairs with mutations: {pairs_with_mutations}")
                    logger.info(f"  - Pairs with suspicious: {pairs_with_suspicious}")

                total_pairs_processed += 1
                original_text1 = sentence1.get('original', '')
                original_text2 = sentence2.get('original', '')

                entities_text1 = extract_entity_texts(original_text1, tagger)
                entities_text2 = extract_entity_texts(original_text2, tagger)

                jaccard_score = jaccard_similarity(entities_text1, entities_text2)

                jaccard_file.write(f"Jaccard Similarity: {jaccard_score:.4f}\n")
                jaccard_file.write(f"Sentence 1: {original_text1}\n")
                jaccard_file.write(f"Sentence 2: {original_text2}\n")
                jaccard_file.write("\n" + "-" * 80 + "\n\n")

                if jaccard_score > JACCARD_THRESHOLD:
                    logger.debug(f"Skipping combination: Jaccard similarity {jaccard_score:.4f} > threshold")
                    pairs_filtered_by_jaccard += 1
                    continue

                mutation_start_time = time.time()

                if ENABLE_ENTITY_FILTER:
                    mutated_results_2, suspicious_results_2 = mutate_and_verify_rule_two(
                        original_text1, original_text2, tagger, NER_MODEL_NAME
                    )
                else:
                    mutated_results_2, suspicious_results_2 = mutate_and_verify_rule_two(
                        original_text1, original_text2, tagger
                    )

                mutation_time = time.time() - mutation_start_time

                if mutated_results_2:
                    pairs_with_mutations += 1
                if suspicious_results_2:
                    pairs_with_suspicious += 1

                if suspicious_results_2:
                    logger.info(f"Found {len(suspicious_results_2)} suspicious results for sentence pair")
                    if len(suspicious_results_2) > 0:
                        sample = suspicious_results_2[0]
                        if "reasons" in sample and sample["reasons"]:
                            logger.info(f"Sample suspicious reason: {sample['reasons'][0]}")
                        elif "original_sentence" in sample:
                            logger.info(f"Sample suspicious sentence: {sample['original_sentence'][:50]}...")

                all_mutated_results.extend(mutated_results_2)
                all_suspicious_results.extend(suspicious_results_2)

        logger.info(f"Finished processing all sentence pairs. Total suspicious results: {len(all_suspicious_results)}")
        logger.info(f"Summary:")
        logger.info(f"  - Total pairs processed: {total_pairs_processed}")
        logger.info(f"  - Pairs filtered by Jaccard: {pairs_filtered_by_jaccard}")
        logger.info(f"  - Pairs with mutations: {pairs_with_mutations}")
        logger.info(f"  - Pairs with suspicious results: {pairs_with_suspicious}")

    except Exception as e:
        logger.error(f"Error processing sentence pairs: {e}", exc_info=True)
        return

    mutated_results_filename = f'{output_dir}/mutated_results2.json'
    suspicious_results_filename = f'{output_dir}/suspicious_sentences2.json'

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

    num_suspicious_sentences = len(all_suspicious_results)
    num_suspicious_entities = sum(len(susp.get("reasons", [])) for susp in all_suspicious_results)
    logger.info(f"Total suspicious sentences: {num_suspicious_sentences}")
    logger.info(f"Total suspicious entities: {num_suspicious_entities}")
    logger.info(f"Mutation results saved to '{mutated_results_filename}'")
    logger.info(f"Suspicious sentences saved to '{suspicious_results_filename}'")
    logger.info(f"Jaccard similarity results saved to '{jaccard_file_path}'")
    logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time / 60:.2f} minutes)")


if __name__ == "__main__":
    # Add command line argument support but keep backward compatibility
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Run Rule2 cross-sentence mutation test')
        parser.add_argument('--model', '-m', help='Path to NER model')
        parser.add_argument('--output', '-o', help='Output directory')
        parser.add_argument('--input', '-i', help='Input JSON file')

        args = parser.parse_args()
        process_mutations_rule_two(args.model, args.output, args.input)
    else:
        # Original behavior - use defaults
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f'{DEFAULT_LOG_DIR}/testRule2-{timestamp}.log'

            # Original logging setup
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_filename),
                    logging.StreamHandler()
                ]
            )

            logger = logging.getLogger(__name__)
            logger.info(f"=== Starting testRule2new.py (Log: {log_filename}) ===")
            logger.info(f"Configuration:")
            logger.info(f"  DEFAULT_MODEL_PATH: {DEFAULT_MODEL_PATH}")
            logger.info(f"  DEFAULT_INPUT_FILE: {DEFAULT_INPUT_FILE}")
            logger.info(f"  DEFAULT_OUTPUT_DIR: {DEFAULT_OUTPUT_DIR}")
            logger.info(f"  ENABLE_ENTITY_FILTER: {ENABLE_ENTITY_FILTER}")
            logger.info(f"  NER_MODEL_NAME: {NER_MODEL_NAME}")
            logger.info(f"  JACCARD_THRESHOLD: {JACCARD_THRESHOLD}")
            logger.info(f"  NUM_RANDOM_PAIRS: {NUM_RANDOM_PAIRS}")
            process_mutations_rule_two()
            logger.info("=== Completed testRule2new.py ===")
        except Exception as e:
            logger.error(f"Unhandled exception: {e}", exc_info=True)
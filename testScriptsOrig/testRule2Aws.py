import json
import random
import os
import time
import logging
import datetime
from itertools import combinations
import traceback

#
from AwsUtils.aws_ner_utils import setup_aws_mock, initialize_aws_tagger, ensure_output_dir

#
API_CALL_COUNT = 0


#
class AwsApiFilter(logging.Filter):
    def filter(self, record):
        #
        if 'Response from AWS' in record.getMessage():
            return False
        if 'Request to AWS' in record.getMessage():
            return False
        if 'AWS response' in record.getMessage():
            return False
        if 'credentials=' in record.getMessage():
            return False
        if 'aws_access_key' in record.getMessage():
            return False
        if 'aws_session_token' in record.getMessage():
            return False
        if 'credential_process' in record.getMessage():
            return False
        return True


#
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'TR-AWS/testRule2-{timestamp}.log'

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

#
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
#
file_handler.addFilter(AwsApiFilter())

#
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
#
console_handler.addFilter(AwsApiFilter())

#
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

#
logger = logging.getLogger(__name__)

setup_aws_mock()


def track_api_call(func):
    """"""

    def wrapper(*args, **kwargs):
        global API_CALL_COUNT
        API_CALL_COUNT += 1
        return func(*args, **kwargs)

    return wrapper


#
from rules.mutationRuleTwo import jaccard_similarity, swap_entities_across_sentences, mutate_and_verify_rule_two
#
from rules.mutationRuleTwo import extract_entity_texts as original_extract_entity_texts
from rules.mutationRuleTwo import extract_entities as original_extract_entities


#
@track_api_call
def extract_entity_texts(text, tagger):
    """"""
    logger.debug(f"API call: extract_entity_texts for text: {text[:50]}...")
    return original_extract_entity_texts(text, tagger)


@track_api_call
def extract_entities(sentence, tagger):
    """"""
    logger.debug(f"API call: extract_entities for sentence: {sentence[:50]}...")
    return original_extract_entities(sentence, tagger)


#
import rules.mutationRuleTwo

rules.mutationRuleTwo.extract_entity_texts = extract_entity_texts
rules.mutationRuleTwo.extract_entities = extract_entities


def process_mutations_rule_two_aws():

    global API_CALL_COUNT

    API_CALL_COUNT = 0

    start_time = time.time()
    logger.info(f"Starting process_mutations_rule_two_aws (Log file: {log_filename})")

    logger.info("Initializing AWS tagger")
    tagger = initialize_aws_tagger()
    logger.info("AWS tagger initialized successfully")

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
    jaccard_threshold = 0.5  #
    num_random_pairs = 5000  #

    #
    logger.info("Generating sentence pairs")
    all_sentence_pairs = list(combinations(data, 2))
    logger.info(f"Total possible sentence pairs: {len(all_sentence_pairs)}")

    #
    logger.info(f"Randomly selecting {num_random_pairs} sentence pairs")
    random_pairs = random.sample(all_sentence_pairs, min(num_random_pairs, len(all_sentence_pairs)))

    #
    ensure_output_dir('TR-AWS')

    #
    jaccard_file_path = 'your_jaccard_file_here'
    logger.info(f"Creating Jaccard similarity results file: {jaccard_file_path}")

    try:
        with open(jaccard_file_path, 'w', encoding='utf-8') as jaccard_file:
            #
            logger.info("Starting to process sentence pairs")
            for i, (sentence1, sentence2) in enumerate(random_pairs):
                if i % 100 == 0 and i > 0:  #
                    logger.info(f"Processed {i}/{len(random_pairs)} sentence pairs, API calls so far: {API_CALL_COUNT}")

                original_text1 = sentence1.get('original', '')
                original_text2 = sentence2.get('original', '')

                #
                entities_text1 = extract_entity_texts(original_text1, tagger)
                entities_text2 = extract_entity_texts(original_text2, tagger)

                jaccard_score = jaccard_similarity(entities_text1, entities_text2)

                #
                jaccard_file.write(f"Jaccard Similarity: {jaccard_score:.4f}\n")
                jaccard_file.write("\n" + "-" * 80 + "\n\n")

                #
                if jaccard_score > jaccard_threshold:
                    logger.debug(f"Skipping combination: Jaccard similarity {jaccard_score:.4f} > threshold")
                    continue

                try:
                    mutation_start_time = time.time()
                    #
                    before_mutation_api_calls = API_CALL_COUNT

                    mutated_results_2, suspicious_results_2 = mutate_and_verify_rule_two(original_text1, original_text2,
                                                                                         tagger)

                    mutation_api_calls = API_CALL_COUNT - before_mutation_api_calls
                    mutation_time = time.time() - mutation_start_time
                    logger.debug(
                        f"Mutation processing used {mutation_api_calls} API calls in {mutation_time:.2f} seconds")

                    if suspicious_results_2:
                        logger.info(f"Found {len(suspicious_results_2)} suspicious results for sentence pair")
                        if len(suspicious_results_2) > 0:
                            sample = suspicious_results_2[0]
                            if "reasons" in sample and sample["reasons"]:
                                logger.info(f"Sample suspicious reason: {sample['reasons'][0]}")
                            elif "original_sentence" in sample:
                                logger.info(f"Sample suspicious sentence: {sample['original_sentence'][:50]}...")

                    #
                    all_mutated_results.extend(mutated_results_2)

                    #
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

    #
    mutated_results_filename = 'your_mutated_file_path'
    suspicious_results_filename = 'your_suspicious_file_path'

    #
    logger.info(f"Writing mutated results to file: {mutated_results_filename}")
    try:
        with open(mutated_results_filename, 'w', encoding='utf-8') as f:
            json.dump(all_mutated_results, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved {len(all_mutated_results)} mutated results")
    except Exception as e:
        logger.error(f"Error saving mutated results: {e}")

    #
    logger.info(f"Writing suspicious sentences to file: {suspicious_results_filename}")
    try:
        with open(suspicious_results_filename, 'w', encoding='utf-8') as f:
            json.dump(all_suspicious_results, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved {len(all_suspicious_results)} suspicious results")
    except Exception as e:
        logger.error(f"Error saving suspicious results: {e}")

    #
    execution_time = time.time() - start_time

    #
    num_suspicious_sentences = len(all_suspicious_results)
    num_suspicious_entities = sum(len(susp.get("reasons", [])) for susp in all_suspicious_results)
    logger.info(f"Total suspicious sentences: {num_suspicious_sentences}")
    logger.info(f"Total suspicious entities: {num_suspicious_entities}")
    logger.info(f"Total AWS API calls: {API_CALL_COUNT}")  # 
    logger.info(f"Mutation results saved to '{mutated_results_filename}'")
    logger.info(f"Suspicious sentences saved to '{suspicious_results_filename}'")
    logger.info(f"Jaccard similarity results saved to '{jaccard_file_path}'")
    logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time / 60:.2f} minutes)")


if __name__ == "__main__":
    try:
        logger.info(f"=== Starting testRule2Aws.py (Log: {log_filename}) ===")
        process_mutations_rule_two_aws()
        logger.info("=== Completed testRule2Aws.py ===")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
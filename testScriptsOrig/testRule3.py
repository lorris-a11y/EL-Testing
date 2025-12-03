
import json
import time
import logging
import datetime
import requests
import traceback
import os
from collections import defaultdict
from flair.models import SequenceTagger
from rules.entity_linking import mutate_and_verify_with_knowledge_graph, \
    get_entities
from AwsUtils.aws_ner_utils import disable_all_proxies, restore_proxy_settings


#
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'TR-Mode1/testRule3-{timestamp}.log'

#
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


USE_DBPEDIA = False  #

# 
if USE_DBPEDIA:
    logger.info("Using DBpedia API for entity descriptions")
else:
    logger.info("Using Wikidata API for entity descriptions")

def test_entity_linking():
    #
    start_time = time.time()
    logger.info(f"Starting test_entity_linking (Log file: {log_filename})")

    original_settings = None
    try:
        # Temporarily disable all proxies to avoid conflicts
        original_settings = disable_all_proxies()
        logger.info("Disabled all proxy settings for initialization.")

        # Load NER model
        logger.info("Loading NER model")
        tagger = SequenceTagger.load('models/conll-large.bin')
        logger.info("NER tagger initialized successfully.")

        # Create a session specifically for DBpedia requests
        dbpedia_session = requests.Session()
#-------------------------------------

        #
        #
        try:
            test_response = requests.get("https://www.wikidata.org", timeout=3)
            logger.info("Direct connection successful, not using proxy for API calls")
            #
        except:
            logger.info("Direct connection failed, using proxy for API calls")
            dbpedia_session.proxies = {
                "http": "your_proxy_here",
                "https": "your_proxy_here",
            }


        #
        dbpedia_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        })

        logger.info("Knowledge graph session created with configuration.")


        # Read input data
        logger.info("Reading data from JSON file")
        try:
            with open('your_test_file_here', 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded data: {len(data)} sentences")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return

        all_mutated_results = []
        all_suspicious_results = []

        # Create output directory paths
        mutated_results_file = "TR-Mode1/mutated_results3-test.json"
        suspicious_sentences_file = "TR-Mode1/suspicious_sentences3-test.json"

        # Process each sentence
        processed_count = 0
        error_count = 0
        total_cases = len(data)

        logger.info("Starting sentence processing")
        for idx, sentence_data in enumerate(data):
            if idx % 10 == 0:
                logger.info(f"Processing sentence {idx + 1}/{total_cases}")
                #
                if idx > 0:
                    elapsed_time = time.time() - start_time
                    sentences_per_second = idx / elapsed_time
                    estimated_total_time = total_cases / sentences_per_second
                    estimated_remaining_time = estimated_total_time - elapsed_time
                    logger.info(
                        f"Progress: {idx / total_cases * 100:.1f}% - Est. remaining time: {estimated_remaining_time / 60:.1f} minutes")

            original_text = sentence_data.get('original', '')

            # Check if sentence has any entities
            try:
                entities = get_entities(original_text, tagger)
                has_entities = False

                for entity_type, entity_list in entities.items():
                    if len(entity_list) > 0:  #
                        has_entities = True
                        break

                if not has_entities:
                    print(f"DEBUG - Skipped sentence (no entities found): {original_text[:100]}...")
                    continue

            except Exception as e:
                logger.error(f"Error checking entities in sentence: {original_text[:100]}...")
                logger.error(f"Error details: {str(e)}")
                error_count += 1
                continue

            # Apply mutation and verify
            try:
                #
                mutation_start_time = time.time()

                # Pass the DBpedia session to the mutation function
                mutated_results, suspicious_results = mutate_and_verify_with_knowledge_graph(
                    original_text, tagger, dbpedia_session
                )

                #
                mutation_time = time.time() - mutation_start_time

                # Add results to collections
                all_mutated_results.extend(mutated_results)
                all_suspicious_results.extend(suspicious_results)
                processed_count += 1

                # Log results for current case
                logger.info(
                    f"Generated {len(mutated_results)} mutations and found {len(suspicious_results)} suspicious cases for sentence {idx + 1}")

                # ，
                if suspicious_results:
                    sample = suspicious_results[0]
                    # ，
                    log_sample = {
                        "original_sentence": sample.get("original_sentence", "")[:50] + "...",
                        "reasons_count": len(sample.get("reasons", []))
                    }
                    if sample.get("reasons"):
                        log_sample["first_reason"] = sample.get("reasons")[0]

                    logger.info(f"Sample suspicious result: {log_sample}")

            except Exception as e:
                logger.error(f"Error processing sentence: {original_text[:100]}...")
                logger.error(f"Error details: {str(e)}")
                logger.error(traceback.format_exc())
                error_count += 1

        logger.info(f"Sentence processing completed. {processed_count} sentences processed with {error_count} errors.")

        # Ensure output directory exists
        os.makedirs("../TR-Mode1", exist_ok=True)

        # Write results to files
        logger.info(f"Writing mutated results to file: {mutated_results_file}")
        try:
            with open(mutated_results_file, 'w', encoding='utf-8') as f:
                json.dump(all_mutated_results, f, ensure_ascii=False, indent=4)
            logger.info(f"Successfully saved {len(all_mutated_results)} mutated results")
        except Exception as e:
            logger.error(f"Error saving mutated results: {str(e)}")

        logger.info(f"Writing suspicious sentences to file: {suspicious_sentences_file}")
        try:
            with open(suspicious_sentences_file, 'w', encoding='utf-8') as f:
                json.dump(all_suspicious_results, f, ensure_ascii=False, indent=4)
            logger.info(f"Successfully saved {len(all_suspicious_results)} suspicious results")
        except Exception as e:
            logger.error(f"Error saving suspicious results: {str(e)}")

        # Calculate and print total execution time
        execution_time = time.time() - start_time

        #
        reasons_count = sum(len(susp.get("reasons", [])) for susp in all_suspicious_results)

        # Log statistics with addition of reasons count
        logger.info("\nTest Statistics:")
        logger.info(f"Total sentences processed: {processed_count}")
        logger.info(f"Total errors encountered: {error_count}")
        logger.info(f"Total mutated sentences: {len(all_mutated_results)}")
        logger.info(f"Total suspicious sentences: {len(all_suspicious_results)}")
        logger.info(f"Total suspicious entity reasons: {reasons_count}")
        logger.info(f"Mutation results saved to '{mutated_results_file}'.")
        logger.info(f"Suspicious sentences saved to '{suspicious_sentences_file}'.")
        logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time / 60:.2f} minutes)")

    except Exception as e:
        logger.critical(f"Critical error in test_entity_linking: {str(e)}")
        logger.critical(traceback.format_exc())

    finally:
        # Always restore original proxy settings, even if exceptions occur
        if original_settings is not None:
            restore_proxy_settings(original_settings)
            logger.info("Restored original proxy settings.")


if __name__ == "__main__":
    try:
        logger.info(f"=== Starting testRule3.py (Log: {log_filename}) ===")
        test_entity_linking()
        logger.info("=== Completed testRule3.py ===")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
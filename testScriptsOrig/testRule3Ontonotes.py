import json
import time
import logging
import datetime
import traceback
import os
from collections import defaultdict
from typing import Dict, List, Any
from flair.models import SequenceTagger
from rules.entity_linking4ontonotes import mutate_and_verify_with_knowledge_graph, get_entities
from AwsUtils.aws_ner_utils import disable_all_proxies, restore_proxy_settings
import requests

#
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'TR-Mode2/testRule3-{timestamp}.log'

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


def analyze_entity_statistics(results: List[Dict[str, Any]]) -> Dict[str, int]:
    entity_counts = defaultdict(int)
    for result in results:
        entities = result.get('entities', {})
        for entity_type in entities.keys():
            entity_counts[entity_type] += 1
    return dict(entity_counts)


def test_entity_linking():
    start_time = time.time()
    logger.info(f"Starting Ontonotes entity linking test (Log file: {log_filename})")

    original_settings = None
    try:
        original_settings = disable_all_proxies()
        logger.info("Disabled all proxy settings for initialization.")

        logger.info("Loading Ontonotes NER model...")
        tagger = SequenceTagger.load('../models/ontonotes-large.bin')
        logger.info("NER tagger initialized successfully.")

        #
        dbpedia_session = requests.Session()

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

        logger.info("Reading data from JSON file")
        with open('txt2json/bbc.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded data: {len(data)} sentences")

        all_mutated_results = []
        all_suspicious_results = []
        entity_type_stats = defaultdict(int)
        processed_count = 0
        error_count = 0
        total_cases = len(data)

        logger.info("Starting sentence processing")
        for idx, sentence_data in enumerate(data):
            if idx % 10 == 0:
                logger.info(f"Processing sentence {idx + 1}/{total_cases}")
                if idx > 0:
                    elapsed_time = time.time() - start_time
                    sentences_per_second = idx / elapsed_time
                    estimated_total_time = total_cases / sentences_per_second
                    estimated_remaining_time = estimated_total_time - elapsed_time
                    logger.info(
                        f"Progress: {idx / total_cases * 100:.1f}% - Est. remaining time: {estimated_remaining_time / 60:.1f} minutes")

            original_text = sentence_data.get('original', '')

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

            try:
                mutation_start_time = time.time()
                mutated_results, suspicious_results = mutate_and_verify_with_knowledge_graph(
                    original_text, tagger, dbpedia_session
                )
                mutation_time = time.time() - mutation_start_time

                #
                for result in mutated_results:
                    for entity_type in result.get('entities', {}).keys():
                        entity_type_stats[entity_type] += 1

                all_mutated_results.extend(mutated_results)
                all_suspicious_results.extend(suspicious_results)
                processed_count += 1

                logger.info(
                    f"Generated {len(mutated_results)} mutations and found {len(suspicious_results)} suspicious cases for sentence {idx + 1}")

                if suspicious_results:
                    sample = suspicious_results[0]
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

        #
        os.makedirs("TR-Mode2", exist_ok=True)

        mutated_path = 'TR-Mode2/mutated_results3.json'
        suspicious_path = 'TR-Mode2/suspicious_sentences3.json'
        stats_path = 'TR-Mode2/ontonotes_stats.json'

        logger.info("Writing mutated results...")
        with open(mutated_path, 'w', encoding='utf-8') as f:
            json.dump(all_mutated_results, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved {len(all_mutated_results)} mutated results")

        logger.info("Writing suspicious sentences...")
        with open(suspicious_path, 'w', encoding='utf-8') as f:
            json.dump(all_suspicious_results, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved {len(all_suspicious_results)} suspicious results")

        # 
        reasons_count = sum(len(susp.get("reasons", [])) for susp in all_suspicious_results)

        stats = {
            "total_cases": total_cases,
            "processed_count": processed_count,
            "error_count": error_count,
            "total_mutations": len(all_mutated_results),
            "suspicious_sentences": len(all_suspicious_results),
            "total_suspicious_reasons": reasons_count,
            "entity_type_stats": dict(entity_type_stats)
        }
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)

        execution_time = time.time() - start_time

        logger.info("\nTest Statistics:")
        logger.info(f"Total sentences processed: {processed_count}")
        logger.info(f"Total errors encountered: {error_count}")
        logger.info(f"Total mutated sentences: {len(all_mutated_results)}")
        logger.info(f"Total suspicious sentences: {len(all_suspicious_results)}")
        logger.info(f"Total suspicious entity reasons: {reasons_count}")
        logger.info(f"Mutation results saved to '{mutated_path}'.")
        logger.info(f"Suspicious sentences saved to '{suspicious_path}'.")
        logger.info(f"Statistics saved to '{stats_path}'.")
        logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time / 60:.2f} minutes)")

        print("\n=== Entity Type Statistics ===")
        for entity_type, count in sorted(entity_type_stats.items()):
            print(f"{entity_type}: {count} occurrences")

    except Exception as e:
        logger.critical(f"Critical error in test_entity_linking: {str(e)}")
        logger.critical(traceback.format_exc())

    finally:
        if original_settings is not None:
            restore_proxy_settings(original_settings)
            logger.info("Restored original proxy settings.")


if __name__ == "__main__":
    try:
        logger.info(f"=== Starting testRule3Ontonotes.py (Log: {log_filename}) ===")
        test_entity_linking()
        logger.info("=== Completed testRule3Ontonotes.py ===")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)

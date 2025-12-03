"""Repair Test Script - Unified Entry
"""
import os
from file_processor import run_repair_from_file
import time


def repair_mr1_flair():
    """Repair MR1 using the Flair model"""
    suspicious_file = ''
    output_file = ''
    model_path = ''

    if os.path.exists(suspicious_file):
        print(f"\n{'=' * 50}\nFlair Repair Results for Rule 1 (MR1)\n{'=' * 50}")
        run_repair_from_file(suspicious_file, output_file, model_path, rule_type="MR1", model_type="flair")
    else:
        print(f"Skipping Flair MR1: File {suspicious_file} does not exist")


def repair_mr2_flair():
    """Repair MR2 using the Flair model"""
    suspicious_file = ''
    output_file = ''
    model_path = ''

    if os.path.exists(suspicious_file):
        print(f"\n{'=' * 50}\nFlair Repair Results for Rule 2 (MR2)\n{'=' * 50}")
        run_repair_from_file(suspicious_file, output_file, model_path, rule_type="MR2", model_type="flair")
    else:
        print(f"Skipping Flair MR2: File {suspicious_file} does not exist")


def repair_mr3_flair():
    """Repair MR3 using the Flair model"""
    suspicious_file = ''
    output_file = ''
    model_path = ''

    if os.path.exists(suspicious_file):
        print(f"\n{'=' * 50}\nFlair Repair Results for Rule 3 (MR3)\n{'=' * 50}")
        run_repair_from_file(suspicious_file, output_file, model_path, rule_type="MR3", model_type="flair")
    else:
        print(f"Skipping Flair MR3: File {suspicious_file} does not exist")


def repair_mr1_azure():
    """Repair MR1 using Azure NER"""
    suspicious_file = ''
    output_file = ''

    from AzureUtils.azure_ner_utils import setup_azure_mock, initialize_azure_tagger, disable_all_proxies, \
        restore_proxy_settings

    original_settings = None
    try:
        setup_azure_mock()
        original_settings = disable_all_proxies()
        print("Initializing Azure NER tagger...")
        azure_tagger = initialize_azure_tagger()
        print("Azure tagger initialized successfully")

        if os.path.exists(suspicious_file):
            print(f"\n{'=' * 50}\nAzure Repair Results for Rule 1 (MR1)\n{'=' * 50}")
            run_repair_from_file_with_tagger(suspicious_file, output_file, azure_tagger, rule_type="MR1",
                                             model_type="cloud")
        else:
            print(f"Skipping Azure MR1: File {suspicious_file} does not exist")

    except Exception as e:
        print(f"Azure NER repair error: {e}")
    finally:
        if original_settings:
            restore_proxy_settings(original_settings)


def repair_mr1_aws():
    """Repair MR1 using AWS NER"""
    suspicious_file = ''
    output_file = ''

    from AwsUtils.aws_ner_utils import setup_aws_mock, initialize_aws_tagger, disable_all_proxies, \
        restore_proxy_settings

    original_settings = None
    try:
        setup_aws_mock()
        original_settings = disable_all_proxies()
        print("Initializing AWS NER tagger...")
        aws_tagger = initialize_aws_tagger()
        print("AWS tagger initialized successfully")

        if os.path.exists(suspicious_file):
            print(f"\n{'=' * 50}\nAWS Repair Results for Rule 1 (MR1)\n{'=' * 50}")
            run_repair_from_file_with_tagger(suspicious_file, output_file, aws_tagger, rule_type="MR1",
                                             model_type="cloud")
        else:
            print(f"Skipping AWS MR1: File {suspicious_file} does not exist")

    except Exception as e:
        print(f"AWS NER repair error: {e}")
    finally:
        if original_settings:
            restore_proxy_settings(original_settings)


def repair_mr2_azure():
    """Repair MR2 using Azure NER"""
    suspicious_file = ''
    output_file = ''

    from AzureUtils.azure_ner_utils import setup_azure_mock, initialize_azure_tagger, disable_all_proxies, \
        restore_proxy_settings

    original_settings = None
    try:
        setup_azure_mock()
        original_settings = disable_all_proxies()
        azure_tagger = initialize_azure_tagger()

        if os.path.exists(suspicious_file):
            print(f"\n{'=' * 50}\nAzure Repair Results for Rule 2 (MR2)\n{'=' * 50}")
            run_repair_from_file_with_tagger(suspicious_file, output_file, azure_tagger, rule_type="MR2",
                                             model_type="cloud")
        else:
            print(f"Skipping Azure MR2: File {suspicious_file} does not exist")
    except Exception as e:
        print(f"Azure NER MR2 repair error: {e}")
    finally:
        if original_settings:
            restore_proxy_settings(original_settings)


def repair_mr2_aws():
    """Repair MR2 using AWS NER"""
    suspicious_file = ''
    output_file = ''

    from AwsUtils.aws_ner_utils import setup_aws_mock, initialize_aws_tagger, disable_all_proxies, \
        restore_proxy_settings

    original_settings = None
    try:
        setup_aws_mock()
        original_settings = disable_all_proxies()
        aws_tagger = initialize_aws_tagger()

        if os.path.exists(suspicious_file):
            print(f"\n{'=' * 50}\nAWS Repair Results for Rule 2 (MR2)\n{'=' * 50}")
            run_repair_from_file_with_tagger(suspicious_file, output_file, aws_tagger, rule_type="MR2",
                                             model_type="cloud")
        else:
            print(f"Skipping AWS MR2: File {suspicious_file} does not exist")
    except Exception as e:
        print(f"AWS NER MR2 repair error: {e}")
    finally:
        if original_settings:
            restore_proxy_settings(original_settings)


def repair_mr3_azure():
    """Repair MR3 using Azure NER"""
    suspicious_file = ''
    output_file = ''

    from AzureUtils.azure_ner_utils import setup_azure_mock, initialize_azure_tagger, disable_all_proxies, \
        restore_proxy_settings

    original_settings = None
    try:
        setup_azure_mock()
        original_settings = disable_all_proxies()
        azure_tagger = initialize_azure_tagger()

        if os.path.exists(suspicious_file):
            print(f"\n{'=' * 50}\nAzure Repair Results for Rule 3 (MR3)\n{'=' * 50}")
            run_repair_from_file_with_tagger(suspicious_file, output_file, azure_tagger, rule_type="MR3",
                                             model_type="cloud")
        else:
            print(f"Skipping Azure MR3: File {suspicious_file} does not exist")
    except Exception as e:
        print(f"Azure NER MR3 repair error: {e}")
    finally:
        if original_settings:
            restore_proxy_settings(original_settings)


def repair_mr3_aws():
    """Repair MR3 using AWS NER"""
    suspicious_file = ''
    output_file = ''

    from AwsUtils.aws_ner_utils import setup_aws_mock, initialize_aws_tagger, disable_all_proxies, \
        restore_proxy_settings

    original_settings = None
    try:
        setup_aws_mock()
        original_settings = disable_all_proxies()
        aws_tagger = initialize_aws_tagger()

        if os.path.exists(suspicious_file):
            print(f"\n{'=' * 50}\nAWS Repair Results for Rule 3 (MR3)\n{'=' * 50}")
            run_repair_from_file_with_tagger(suspicious_file, output_file, aws_tagger, rule_type="MR3",
                                             model_type="cloud")
        else:
            print(f"Skipping AWS MR3: File {suspicious_file} does not exist")
    except Exception as e:
        print(f"AWS NER MR3 repair error: {e}")
    finally:
        if original_settings:
            restore_proxy_settings(original_settings)


def run_repair_from_file_with_tagger(suspicious_file: str, output_file: str, tagger, rule_type="default",
                                     model_type="cloud"):
    """
    Perform repair directly using the provided tagger

    Args:
        suspicious_file: Path to the JSON file containing suspicious sentences
        output_file: Path to save the repair results
        tagger: NER tagger
        rule_type: Rule type
        model_type: Model type ("flair" or "cloud")
    """
    from file_processor import process_suspicious_file
    import json

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Process the suspicious file
    print(f"Processing suspicious file: {suspicious_file}")
    fixed_results = process_suspicious_file(suspicious_file, tagger, rule_type, model_type)

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

    print("\nRepair Statistics:")
    print(f"Total sentences processed: {len(fixed_results)}")
    print(f"Total entities attempted to repair: {total_repairs}")

    if total_repairs > 0:
        original_percent = (original_restored / total_repairs * 100)
        print(f"Restored original tags: {original_restored} ({original_percent:.1f}%)")

        mutated_percent = (mutated_kept / total_repairs * 100)
        print(f"Kept mutated tags: {mutated_kept} ({mutated_percent:.1f}%)")

        new_tag_percent = (new_tag / total_repairs * 100)
        print(f"Assigned new tags: {new_tag} ({new_tag_percent:.1f}%)")


if __name__ == "__main__":
    print("=== Starting Entity Repair Tests ===")

    # Repair suspicious results for various rules - Flair version
    # print("\n1. Repair using Flair model:")
    # repair_mr1_flair()
    # repair_mr2_flair()
    # repair_mr3_flair()

    # Repair suspicious results for various rules - Azure version
    # print("\n2. Repair using Azure NER:")
    # repair_mr1_azure()
    # repair_mr2_azure()
    repair_mr3_azure()

    # Repair suspicious results for various rules - AWS version
    # print("\n3. Repair using AWS NER:")
    # repair_mr1_aws()
    # repair_mr2_aws()
    # repair_mr3_aws()

    print("\n=== All Repair Tests Completed ===")
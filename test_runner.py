import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Run mutation tests with different rules and NER models')

    parser.add_argument('rule',
                        choices=['rule1', 'rule2', 'rule3'],
                        help='Mutation rule to test')

    parser.add_argument('model',
                        choices=['conll', 'aws', 'azure', 'ontonotes'],
                        help='NER model to use for testing')

    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        print(f"Running {args.rule.upper()} with {args.model.upper()} model...")

    # Save current working directory
    original_cwd = os.getcwd()

    try:
        if args.rule == 'rule1':
            if args.model == 'conll':
                from testScriptsOrig.testRule1 import process_mutations_rule_one
                os.chdir(original_cwd)
                process_mutations_rule_one()
            elif args.model == 'aws':
                from testScriptsOrig.testRule1_aws import process_mutations_rule_one_aws
                os.chdir(original_cwd)
                process_mutations_rule_one_aws()
            elif args.model == 'azure':
                from testScriptsOrig.testRule1_Azure import process_mutations_rule_one_azure
                os.chdir(original_cwd)
                process_mutations_rule_one_azure()
            elif args.model == 'ontonotes':
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                from testScriptsOrig.testRule1_ontonotes import process_mutations_rule_one
                os.chdir(original_cwd)
                process_mutations_rule_one()

        elif args.rule == 'rule2':
            if args.model == 'conll':
                from testScriptsOrig.testRule2new import process_mutations_rule_two
                os.chdir(original_cwd)
                process_mutations_rule_two(
                    model_path='models/conll-large.bin',
                    output_dir='TR-Mode1',
                    input_file='txt2json/bbcNews2.json'
                )
            elif args.model == 'ontonotes':
                from testScriptsOrig.testRule2new import process_mutations_rule_two
                os.chdir(original_cwd)
                process_mutations_rule_two(
                    model_path='models/ontonotes-large.bin',
                    output_dir='TR-Mode2',
                    input_file='txt2json/bbc.json'
                )
            elif args.model == 'aws':
                from testScriptsOrig.testRule2Aws import process_mutations_rule_two_aws
                os.chdir(original_cwd)
                process_mutations_rule_two_aws()
            elif args.model == 'azure':
                from testScriptsOrig.testRule2Azure import process_mutations_rule_two_azure
                os.chdir(original_cwd)
                process_mutations_rule_two_azure()

        elif args.rule == 'rule3':
            if args.model == 'conll':
                from testScriptsOrig.testRule3 import test_entity_linking
                os.chdir(original_cwd)
                test_entity_linking()
            elif args.model == 'aws':
                from testScriptsOrig.testRule3Aws import process_mutations_rule_three_aws
                os.chdir(original_cwd)
                process_mutations_rule_three_aws()
            elif args.model == 'azure':
                from testScriptsOrig.testRule3Azure import process_mutations_rule_three_azure
                os.chdir(original_cwd)
                process_mutations_rule_three_azure()
            elif args.model == 'ontonotes':
                from testScriptsOrig.testRule3Ontonotes import test_entity_linking
                os.chdir(original_cwd)
                test_entity_linking()

        print(f"✓ {args.rule.upper()}-{args.model.upper()} test completed successfully!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Execution error: {e}")
        sys.exit(1)

    finally:
        # Restore original working directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()
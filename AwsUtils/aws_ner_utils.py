import os
import sys
import types
import boto3


# Function to disable all proxies
def disable_all_proxies():
    """Disable all proxy settings"""
    # Record original settings
    original_settings = {}

    # Environment variable proxy settings
    proxy_env_vars = [
        'HTTP_PROXY', 'http_proxy',
        'HTTPS_PROXY', 'https_proxy',
        'FTP_PROXY', 'ftp_proxy',
        'NO_PROXY', 'no_proxy',
        'ALL_PROXY', 'all_proxy'
    ]

    for var in proxy_env_vars:
        if var in os.environ:
            original_settings[var] = os.environ[var]
            del os.environ[var]

    # Set no_proxy to all
    os.environ['NO_PROXY'] = '*'

    return original_settings


# Restore original proxy settings
def restore_proxy_settings(original_settings):
    """Restore original proxy settings"""
    # Restore environment variables
    for var, value in original_settings.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


# AWS Sentence and Span classes
class AWSSentence:
    def __init__(self, text):
        self.text = text
        self.spans = {}

    def get_spans(self, layer_name):
        if layer_name in self.spans:
            return self.spans[layer_name]
        return []


class AWSEntitySpan:
    def __init__(self, text, start_pos, end_pos, tag):
        self.text = text
        self.start_position = start_pos
        self.end_position = end_pos
        self.tag = tag


# AWS Tagger
class AWSTagger:
    def __init__(self, client=None):
        self.client = client or boto3.client('comprehend')

    def predict(self, sentence):
        text = sentence.text

        try:
            response = self.client.detect_entities(
                Text=text,
                LanguageCode='en'  # 根据需要修改语言代码
            )

            print(f"\nEntities returned by AWS Comprehend API ('{text[:50]}...'):")

            entities = []
            if 'Entities' not in response or not response['Entities']:
                print("No entities found by AWS Comprehend")
            else:
                for entity in response['Entities']:
                    print(f"  {entity['Type']}: '{entity['Text']}' ({entity['BeginOffset']}, {entity['EndOffset']})")
                    span = AWSEntitySpan(
                        text=entity['Text'],
                        start_pos=entity['BeginOffset'],
                        end_pos=entity['EndOffset'],
                        tag=entity['Type']
                    )
                    entities.append(span)

            sentence.spans = {'ner': entities}

        except Exception as e:
            print(f"AWS Comprehend NER API error: {str(e)}")
            sentence.spans = {'ner': []}


def setup_aws_mock():
    """Setup mock Flair module to use AWS classes"""
    sys.modules['flair.data'] = types.ModuleType('flair.data')
    sys.modules['flair.data'].Sentence = AWSSentence
    sys.modules['flair.models'] = types.ModuleType('flair.models')
    sys.modules['flair.models'].SequenceTagger = AWSTagger


def initialize_aws_tagger():
    """Initialize and return an AWS tagger instance with proxy handling."""
    # Completely disable all proxy settings
    original_settings = disable_all_proxies()

    try:
        # Output current proxy environment variables for debugging
        print("Current proxy environment variables:")
        for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY']:
            print(f"  {var}: {os.environ.get(var, 'Not set')}")

        # Create AWS client with no proxies
        client = boto3.client('comprehend')
        tagger = AWSTagger(client)
        print("AWS Comprehend client initialized successfully")
        return tagger

    except Exception as e:
        print(f"Error initializing AWS tagger: {str(e)}")
        raise

    finally:
        # Restore original proxy settings
        restore_proxy_settings(original_settings)


# Helper function to create output directories
def ensure_output_dir(directory):
    """Ensure the output directory exists."""
    os.makedirs(directory, exist_ok=True)
    return directory



if __name__ == "__main__":

    setup_aws_mock()


    tagger = initialize_aws_tagger()


    sentence = AWSSentence("Jeff Bezos founded Amazon in 1994 and opened headquarters in Seattle.")

    tagger.predict(sentence)

    for entity_span in sentence.get_spans('ner'):
        print(f"Entity: {entity_span.text}, Type: {entity_span.tag}, "
              f"Position: ({entity_span.start_position}, {entity_span.end_position})")
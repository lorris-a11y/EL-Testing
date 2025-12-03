
import os
import sys
import types
import socket
import requests.utils
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv


def disable_all_proxies():

    original_settings = {}

    proxy_env_vars = [
        'HTTP_PROXY', 'http_proxy',
        'HTTPS_PROXY', 'https_proxy',
        'FTP_PROXY', 'ftp_proxy',
        'NO_PROXY', 'no_proxy',
        'ALL_PROXY', 'all_proxy',
        'AZURE_HTTP_USER_AGENT'
    ]

    for var in proxy_env_vars:
        if var in os.environ:
            original_settings[var] = os.environ[var]
            del os.environ[var]

    original_settings['requests_proxies'] = requests.utils.getproxies()
    if hasattr(requests.utils, 'proxies'):
        requests.utils.proxies = {}

    os.environ['NO_PROXY'] = '*'

    return original_settings


def restore_proxy_settings(original_settings):
    """恢复原始代理设置"""
    # 恢复环境变量
    for var, value in original_settings.items():
        if var != 'requests_proxies':
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    if 'requests_proxies' in original_settings and hasattr(requests.utils, 'proxies'):
        requests.utils.proxies = original_settings['requests_proxies']


# Azure Sentence and Span classes
class AzureSentence:
    def __init__(self, text):
        self.text = text
        self.spans = {}

    def get_spans(self, layer_name):
        if layer_name in self.spans:
            return self.spans[layer_name]
        return []


class AzureEntitySpan:
    def __init__(self, text, start_pos, end_pos, tag, score):
        self.text = text
        self.start_position = start_pos
        self.end_position = end_pos
        self.tag = tag
        self.score = score


# Azure Tagger
class AzureTagger:
    def __init__(self, client):
        self.client = client

    def predict(self, sentence):
        text = sentence.text
        documents = [text]

        try:
            response = self.client.recognize_entities(documents=documents)
            if not response:
                print(f"Azure API returned empty result")
                sentence.spans = {'ner': []}
                return

            result = response[0]
            if not hasattr(result, 'entities'):
                print("No entities attribute in Azure result")
                sentence.spans = {'ner': []}
                return

            entities = []
            if not result.entities:
                print("No entities found by Azure")
            else:
                print(f"\nEntities returned by Azure API ('{text[:50]}...'):")
                for entity in result.entities:
                    print(f"  {entity.category}: '{entity.text}' ({entity.offset}, {entity.offset + entity.length})")
                    span = AzureEntitySpan(
                        text=entity.text,
                        start_pos=entity.offset,
                        end_pos=entity.offset + entity.length,
                        tag=entity.category,
                        score=entity.confidence_score
                    )
                    entities.append(span)

            sentence.spans = {'ner': entities}

        except Exception as e:
            print(f"Azure NER API error: {str(e)}")
            sentence.spans = {'ner': []}


def setup_azure_mock():
    """Setup mock Flair module to use Azure classes"""
    sys.modules['flair.data'] = types.ModuleType('flair.data')
    sys.modules['flair.data'].Sentence = AzureSentence
    sys.modules['flair.models'] = types.ModuleType('flair.models')
    sys.modules['flair.models'].SequenceTagger = AzureTagger


def authenticate_client():
    """Authenticate the Azure client using environment variables with stronger proxy disabling."""

    original_settings = disable_all_proxies()

    try:

        load_dotenv()
        language_key = os.getenv('LANGUAGE_KEY')
        language_endpoint = os.getenv('LANGUAGE_ENDPOINT')

        if not language_key or not language_endpoint:
            raise ValueError("Azure credentials not found in environment variables. "
                             "Please set LANGUAGE_KEY and LANGUAGE_ENDPOINT.")


        print("Current proxy environment variables:")
        for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY']:
            print(f"  {var}: {os.environ.get(var, 'Not set')}")
        print(f"  requests.utils.getproxies(): {requests.utils.getproxies()}")


        ta_credential = AzureKeyCredential(language_key)


        text_analytics_client = TextAnalyticsClient(
            endpoint=language_endpoint,
            credential=ta_credential,
            connection_timeout=30,
            read_timeout=30
        )


        print("Azure client initialized successfully")

        return text_analytics_client

    except Exception as e:
        print(f"Error authenticating Azure client: {str(e)}")
        raise

    finally:

        restore_proxy_settings(original_settings)


def initialize_azure_tagger():
    """Initialize and return an Azure tagger instance with stronger proxy disabling."""

    original_settings = disable_all_proxies()

    try:

        client = authenticate_client()
        tagger = AzureTagger(client)
        return tagger

    except Exception as e:
        print(f"Error initializing Azure tagger: {str(e)}")
        raise

    finally:

        restore_proxy_settings(original_settings)


# Helper function to create output directories
def ensure_output_dir(directory):
    """Ensure the output directory exists."""
    os.makedirs(directory, exist_ok=True)
    return directory
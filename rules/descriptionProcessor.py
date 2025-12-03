import re
from functools import lru_cache
from typing import Optional

import requests

#  "wikidata_only", "wikipedia_only"
API_STRATEGY = "wikidata_only"


class EnhancedWikidataProcessor:
    """"""

    def __init__(self, session=None):
        self.session = session or requests.Session()
        self.wikidata_api_url = "https://www.wikidata.org/w/api.php"
        self.wikidata_sparql_url = "https://query.wikidata.org/sparql"


        self.type_to_wikidata_class = {
            "PERSON": ["Q5"],  # human

            "ORGANIZATION": ["Q43229", "Q4830453", "Q783794"],  # organization, business enterprise, company

            "LOCATION": [
                "Q2221906",  # geographic location (general)
                "Q515",  # city
                "Q6256",  # country (sovereign state)
                "Q3336843",  # country of the United Kingdom
                "Q3024240",  # historical country
                "Q7275",  # state
                "Q35657",  # U.S. state
                "Q10864048",  # constituent country
                "Q82794",  # geographic region
                "Q56061",  # administrative territorial entity
                "Q13226383",  # facility (from FAC → LOCATION)
                "Q811979",  # architectural structure (from FAC → LOCATION)
                "Q41176",  # building (from FAC → LOCATION)
                "Q6999",
            ],

            "EVENT": ["Q1190554", "Q1656682"],  # occurrence, event

            "WORK_OF_ART": ["Q838948", "Q47461344", "Q11424", "Q134556", "Q7889"],
            # work of art, written work, film, single, video game

            "PRODUCT": ["Q2424752", "Q235557", "Q15401930"],  # product, manufactured good, brand

            "GROUP": ["Q41710", "Q179805", "Q7278", "Q6957273"],
            # ethnic group, religious group, political party, nationality (from NORP → GROUP)

            "MISC": ["Q35120"],  # entity (generic)
        }

    def get_entity_with_type_verification(self, entity_text: str, std_entity_type: str):
        """"""
        try:
            # 1.
            response = self.session.get(
                self.wikidata_api_url,
                headers={'Accept': 'application/json'},
                params={
                    'action': 'wbsearchentities',
                    'search': entity_text,
                    'language': 'en',
                    'format': 'json',
                    'limit': 8,
                    'type': 'item'
                },
                timeout=15
            )

            if response.status_code != 200:
                return None

            data = response.json()
            search_results = data.get("search", [])

            if not search_results:
                return None

            # 2.
            for candidate in search_results:
                entity_id = candidate.get('id')
                if self._verify_entity_type(entity_id, std_entity_type):
                    return {
                        'id': entity_id,
                        'label': candidate.get('label'),
                        'description': candidate.get('description'),
                        'verified_type': std_entity_type
                    }

            # 3. ，
            return {
                'id': search_results[0].get('id'),
                'label': search_results[0].get('label'),
                'description': search_results[0].get('description'),
                'verified_type': None
            }

        except Exception as e:
            print(f"Error in enhanced Wikidata lookup: {e}")
            return None

    def _verify_entity_type(self, entity_id: str, expected_type: str) -> bool:
        """"""

        #
        if expected_type == "MISC":
            print(f"DEBUG - Skipping type verification for MISC entity {entity_id}")
            return True

        expected_classes = self.type_to_wikidata_class.get(expected_type, [])
        if not expected_classes:
            return False

        #
        class_filter = " || ".join([f"?type = wd:{cls}" for cls in expected_classes])

        query = f"""
        SELECT ?type WHERE {{
            wd:{entity_id} wdt:P31/wdt:P279* ?type .
            FILTER({class_filter})
        }} LIMIT 1
        """

        try:
            response = self.session.get(
                self.wikidata_sparql_url,
                params={'query': query, 'format': 'json'},
                headers={
                    'Accept': 'application/json',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                },
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                return len(data.get('results', {}).get('bindings', [])) > 0

        except Exception as e:
            print(f"SPARQL verification failed: {e}")

        return False

#====================================================



class MultiNEREntityProcessor:
    """"""

    def __init__(self, session=None):
        self.session = session or requests.Session()
        self.wikipedia_api_url = "https://en.wikipedia.org/api/rest_v1/page/summary"
        self.wikidata_api_url = "https://www.wikidata.org/w/api.php"
        #
        self.api_strategy = API_STRATEGY
        #
        self.enhanced_wikidata_processor = EnhancedWikidataProcessor(session)

    def get_entity_description(self, entity_text: str, entity_type: str, ner_model: str = "conll") -> str:
        """"""
        try:
            std_entity_type = self._standardize_entity_type(entity_type, ner_model)
            print(
                f"\nProcessing entity: {entity_text} ({entity_type} -> {std_entity_type}) [Model: {ner_model}] [Strategy: {self.api_strategy}]")

            if self.api_strategy == "wikidata_only":
                return self._get_wikidata_only_description(entity_text, std_entity_type, entity_type, ner_model)

            elif self.api_strategy == "wikipedia_only":
                return self._get_wikipedia_only_description(entity_text, std_entity_type, entity_type, ner_model)

            else:
                #
                print(f"Warning: Unknown strategy '{self.api_strategy}', defaulting to 'wikidata_only'")
                return self._get_wikidata_only_description(entity_text, std_entity_type, entity_type, ner_model)

        except Exception as e:
            print(f"✗ Error in description process: {e}")
            return self._get_fallback_description(entity_type, ner_model)



    def _get_wikidata_only_description(self, entity_text: str, std_entity_type: str, entity_type: str,
                                       ner_model: str) -> str:
        """"""
        #
        result = self.enhanced_wikidata_processor.get_entity_with_type_verification(
            entity_text, std_entity_type
        )

        if result:
            if result['verified_type']:
                #
                description = result['description']
                if description:
                    processed = self._process_wikidata_description(description, entity_text, std_entity_type)
                    if processed:
                        print(f"✓ Using type-verified Wikidata description: {processed}")
                        return processed
            else:
                # ，
                print(f"Type mismatch for {entity_text}, using fallback")

        fallback = self._get_fallback_description(entity_type, ner_model)
        print(f"! Using fallback description: {fallback}")
        return fallback



    def _get_wikipedia_only_description(self, entity_text: str, std_entity_type: str, entity_type: str,
                                        ner_model: str) -> str:
        """ """
        wiki_description = self._get_wikipedia_description(entity_text)
        if wiki_description:
            processed = self._process_description(wiki_description, entity_text, std_entity_type)
            if processed:
                print(f"✓ Using Wikipedia description: {processed}")
                return processed

        fallback = self._get_fallback_description(entity_type, ner_model)
        print(f"! Using fallback description: {fallback}")
        return fallback



    def _standardize_entity_type(self, entity_type: str, ner_model: str) -> str:
        """"""

        if ner_model.lower() == "conll3":
            # CoNLL-2003 NER (4)：PER, ORG, LOC, MISC
            conll3_mapping = {
                "PER": "PERSON",
                "ORG": "ORGANIZATION",
                "LOC": "LOCATION",
                "MISC": "MISC"
            }
            return conll3_mapping.get(entity_type, "MISC")

        elif ner_model.lower() == "ontonotes":
            # OntoNotes 5.0 (18)
            ontonotes_mapping = {
                "PERSON": "PERSON",
                "ORG": "ORGANIZATION",
                "GPE": "LOCATION",  #
                "LOC": "LOCATION",
                "FAC": "LOCATION",  #
                "EVENT": "EVENT",
                "WORK_OF_ART": "WORK_OF_ART",
                "PRODUCT": "PRODUCT",
                "NORP": "GROUP",  #
                "LANGUAGE": "MISC",
                "DATE": "DATE",
                "TIME": "TIME",
                "PERCENT": "QUANTITY",
                "MONEY": "QUANTITY",
                "QUANTITY": "QUANTITY",
                "ORDINAL": "QUANTITY",
                "CARDINAL": "QUANTITY",
                "LAW": "MISC"
            }
            return ontonotes_mapping.get(entity_type, "MISC")

        elif ner_model.lower() == "azure":
            # Azure Text Analytics NER
            azure_mapping = {
                "Person": "PERSON",
                "PersonType": "PERSON",
                "Location": "LOCATION",
                "Organization": "ORGANIZATION",
                "Event": "EVENT",
                "Product": "PRODUCT",
                "Skill": "MISC",
                "Address": "LOCATION",
                "PhoneNumber": "CONTACT",
                "Email": "CONTACT",
                "URL": "CONTACT",
                "IP": "CONTACT",
                "DateTime": "DATE",
                "Quantity": "QUANTITY"
            }
            return azure_mapping.get(entity_type, "MISC")

        elif ner_model.lower() == "aws":
            # AWS Comprehend NER
            aws_mapping = {
                "PERSON": "PERSON",
                "LOCATION": "LOCATION",
                "ORGANIZATION": "ORGANIZATION",
                "COMMERCIAL_ITEM": "PRODUCT",
                "EVENT": "EVENT",
                "DATE": "DATE",
                "QUANTITY": "QUANTITY",
                "TITLE": "WORK_OF_ART",
                "OTHER": "MISC"
            }
            return aws_mapping.get(entity_type, "MISC")

        #
        return entity_type

    @lru_cache(maxsize=200)
    def _get_wikipedia_description(self, entity_text: str) -> Optional[str]:
        """"""
        try:
            response = self.session.get(
                f"{self.wikipedia_api_url}/{entity_text.replace(' ', '_')}",
                headers={'Accept': 'application/json'},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('extract', '')

        except Exception as e:
            print(f"Wikipedia API error: {e}")

        return None


    @lru_cache(maxsize=200)
    def _get_wikidata_description(self, entity_text: str, std_entity_type: str) -> Optional[str]:
        """"""
        try:
            #
            result = self.enhanced_wikidata_processor.get_entity_with_type_verification(
                entity_text, std_entity_type
            )

            if result:
                #
                if result['verified_type']:
                    print(f"✓ Type-verified Wikidata entity: {result['label']} - {result['description']}")
                    return result['description']
                else:
                    #
                    print(f"⚠ Type mismatch for {entity_text}: expected {std_entity_type}, got {result['description']}")
                    #
                    # return None
                    return result['description']

            return None

        except Exception as e:
            print(f"Wikidata API error: {e}")
            return None



    #
    def _process_description(self, raw_description: str, entity_name: str, std_entity_type: str) -> str:

        if not raw_description:
            return ""

        # 1.
        cleaned = self._basic_cleaning(raw_description)

        # 2.
        first_sentence = self._extract_first_sentence(cleaned)
        if not first_sentence:
            return ""

        # 3.
        core_identity = self._extract_core_identity(first_sentence, entity_name)
        if not core_identity:
            return ""

        # 4.
        return self._format_final_description(entity_name, core_identity)



    def _process_wikidata_description(self, raw_description: str, entity_name: str, std_entity_type: str) -> str:
        """"""
        if not raw_description:
            return ""

        #
        cleaned = raw_description.strip()

        #
        if cleaned and not cleaned.lower().startswith(('a ', 'an ', 'the ')):
            if cleaned[0].lower() in 'aeiou':
                cleaned = f"an {cleaned}"
            else:
                cleaned = f"a {cleaned}"

        return self._format_final_description(entity_name, cleaned)

    def _basic_cleaning(self, text: str) -> str:
        """"""
        if not text:
            return ""

        #
        text = re.sub(r'<[^>]+>', '', text)

        #
        text = re.sub(r'\[\d+\]', '', text)

        #
        text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)
        text = re.sub(r'\([^)]*born[^)]*\)', '', text, flags=re.IGNORECASE)

        #
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _extract_first_sentence(self, text: str) -> str:
        """"""
        if not text:
            return ""

        #
        sentences = re.split(r'\.(?=\s+[A-Z])', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  #
                return sentence

        return sentences[0] if sentences else ""


    def _extract_core_identity(self, sentence: str, entity_name: str) -> str:
        """"""
        if not sentence:
            return ""

        #
        sentence_without_name = sentence
        if entity_name in sentence:
            sentence_without_name = sentence.replace(entity_name, "", 1).strip()

        #
        patterns = [
            # "is a/an [identity]"
            r'(?:is|was)\s+(a|an)\s+([^,\.]+?)(?=\s+(?:who|which|that|and|known|\.|,|$))',
            # "is [identity]" (without a/an)
            r'(?:is|was)\s+([^,\.]+?)(?=\s+(?:who|which|that|and|known|\.|,|$))',
        ]

        for pattern in patterns:
            match = re.search(pattern, sentence_without_name, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    #
                    article = match.group(1)
                    identity = match.group(2).strip()
                    return f"{article} {identity}"
                else:
                    #
                    identity = match.group(1).strip()

                    #
                    if identity.lower().startswith(('a ', 'an ', 'the ')):
                        return identity  #

                    # 智能添加冠词
                    if identity and identity[0].lower() in 'aeiou':
                        return f"an {identity}"
                    else:
                        return f"a {identity}"

        return ""

    def _format_final_description(self, entity_name: str, core_identity: str) -> str:
        """"""
        if not core_identity:
            return ""

        #
        description = f"{entity_name} is {core_identity}"

        #
        if not description.endswith('.'):
            description += '.'

        return description



    def _get_fallback_description(self, entity_type: str, ner_model: str) -> str:
        """"""

        #
        fallback_mappings = {
            "conll3": {
                "PER": "a person",
                "ORG": "an organization",
                "LOC": "a location",
                "MISC": "an entity"
            },
            "ontonotes": {
                "PERSON": "a person",
                "ORG": "an organization",
                "GPE": "a place",  #
                "LOC": "a location",
                "FAC": "a facility",  #
                "EVENT": "an event",
                "WORK_OF_ART": "a work of art",
                "PRODUCT": "a product",
                "NORP": "a group",  #
                "LANGUAGE": "a language",
                "LAW": "a law",
                "DATE": "a date",
                "TIME": "a time",
                "PERCENT": "a percentage",
                "MONEY": "an amount of money",
                "QUANTITY": "a quantity",
                "ORDINAL": "an ordinal number",
                "CARDINAL": "a number"
            },
            "azure": {
                "Person": "a person",
                "PersonType": "a person type",
                "Organization": "an organization",
                "Location": "a location",
                "Event": "an event",
                "Product": "a product",
                "Skill": "a skill",
                "Address": "an address",
                "PhoneNumber": "a phone number",
                "Email": "an email address",
                "URL": "a web address",
                "IP": "an IP address",
                "DateTime": "a date and time",
                "Quantity": "a quantity"
            },
            "aws": {
                "PERSON": "a person",
                "ORGANIZATION": "an organization",
                "LOCATION": "a location",
                "COMMERCIAL_ITEM": "a product",
                "EVENT": "an event",
                "DATE": "a date",
                "QUANTITY": "a quantity",
                "TITLE": "a creative work",
                "OTHER": "an entity"
            }
        }

        #
        model_fallbacks = fallback_mappings.get(ner_model.lower(), {})

        #
        return model_fallbacks.get(entity_type, "an entity")


#
def get_entity_description_conll3(entity_text: str, entity_type: str, session=None) -> str:
    """CoNLL-2003 """
    processor = MultiNEREntityProcessor(session)
    return processor.get_entity_description(entity_text, entity_type, "conll3")


def get_entity_description_ontonotes(entity_text: str, entity_type: str, session=None) -> str:
    """OntoNotes """
    processor = MultiNEREntityProcessor(session)
    return processor.get_entity_description(entity_text, entity_type, "ontonotes")


def get_entity_description_azure(entity_text: str, entity_type: str, session=None) -> str:
    """Azure """
    processor = MultiNEREntityProcessor(session)
    return processor.get_entity_description(entity_text, entity_type, "azure")


def get_entity_description_aws(entity_text: str, entity_type: str, session=None) -> str:
    """AWS """
    processor = MultiNEREntityProcessor(session)
    return processor.get_entity_description(entity_text, entity_type, "aws")


#
def get_entity_description_multi(entity_text: str, entity_type: str, ner_model: str, session=None) -> str:
    """

    """
    processor = MultiNEREntityProcessor(session)
    return processor.get_entity_description(entity_text, entity_type, ner_model)



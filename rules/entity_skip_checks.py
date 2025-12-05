"""

"""

import re
import spacy

from .constants import HONORIFICS_SET

try:
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    print("DEBUG - spaCy model loaded for entity skip checks")
except:
    nlp = None
    print("Warning: spaCy model 'en_core_web_sm' not loaded for entity skip checks.")

#
def has_spacy_noun_before_ner_entity(sentence: str, entity: tuple) -> bool:

    entity_text, start_pos, end_pos = entity

    try:
        # 
        doc = nlp(sentence)

        # 
        text_before_entity = sentence[:start_pos]

        if not text_before_entity.strip():
            return False

        # 
        last_meaningful_token = None

        for token in doc:
            # 
            if token.idx + len(token.text) <= start_pos:
                # 
                if not token.is_punct and not token.is_space:
                    if token.text.lower().rstrip('.') not in HONORIFICS_SET:  # 使用导入的常量
                        last_meaningful_token = token

        # 
        if last_meaningful_token and last_meaningful_token.pos_ in ['NOUN', 'PROPN']:
            print(
                f"DEBUG - spaCy found {last_meaningful_token.pos_} '{last_meaningful_token.text}' before NER entity '{entity_text}' (identified by your tagger)")
            return True

        return False

    except Exception as e:
        print(f"DEBUG - spaCy analysis failed: {e}")
        return False


def has_pronoun_before_entity_spacy(sentence: str, entity: tuple) -> bool:
    entity_text, start_pos, end_pos = entity

    try:
        doc = nlp(sentence)

        text_before_entity = sentence[:start_pos]

        if not text_before_entity.strip():
            return False

        last_meaningful_token = None

        for token in doc:
            if token.idx + len(token.text) <= start_pos:
                if not token.is_punct and not token.is_space:
                    last_meaningful_token = token

        if last_meaningful_token and last_meaningful_token.pos_ == 'PRON':
            print(
                f"DEBUG - spaCy found pronoun '{last_meaningful_token.text}' before NER entity '{entity_text}' - skipping to avoid pronoun+pronoun pattern")
            return True

        return False

    except Exception as e:
        print(f"DEBUG - spaCy pronoun analysis failed: {e}")
        return False

def has_spacy_noun_adjacent_to_ner_entity(sentence: str, entity: tuple) -> bool:
    """
    """
    entity_text, start_pos, end_pos = entity

    try:
        doc = nlp(sentence)

        text_before_entity = sentence[:start_pos]
        if text_before_entity.strip():
            last_meaningful_token = None

            for token in doc:
                if token.idx + len(token.text) <= start_pos:
                    if not token.is_punct and not token.is_space:
                        honorifics_set = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'dame', 'lord', 'lady'}
                        if token.text.lower().rstrip('.') not in honorifics_set:
                            last_meaningful_token = token

            if last_meaningful_token and last_meaningful_token.pos_ in ['NOUN', 'PROPN']:
                print(
                    f"DEBUG - spaCy found {last_meaningful_token.pos_} '{last_meaningful_token.text}' before NER entity '{entity_text}'")
                return True

        text_after_entity = sentence[end_pos:]
        if text_after_entity.strip():
            first_meaningful_token = None

            for token in doc:
                if token.idx >= end_pos:
                    if not token.is_punct and not token.is_space:
                        first_meaningful_token = token
                        break

            if (first_meaningful_token and
                    first_meaningful_token.pos_ == 'NOUN' and  # 
                    not first_meaningful_token.ent_type_):  # 

                print(
                    f"DEBUG - spaCy found NOUN '{first_meaningful_token.text}' after NER entity '{entity_text}'")
                return True

        return False

    except Exception as e:
        print(f"DEBUG - spaCy noun adjacency analysis failed: {e}")
        return False


def is_entity_in_quotes(sentence: str, entity: tuple) -> bool:

    entity_text, start_pos, end_pos = entity

    text_before = sentence[:start_pos]
    text_after = sentence[end_pos:]

    quote_chars = ['"', "'", '"', '"', ''', ''']

    for quote in quote_chars:
        if quote in text_before and quote in text_after:
            last_quote_before = text_before.rfind(quote)
            first_quote_after = text_after.find(quote)

            if last_quote_before != -1 and first_quote_after != -1:
                quoted_content = sentence[last_quote_before + 1:end_pos + first_quote_after]
                if entity_text in quoted_content and len(quoted_content.strip()) <= len(entity_text) + 10:
                    print(f"DEBUG - Skipping entity in quotes: '{entity_text}' appears to be quoted")
                    return True

    return False


def is_entity_in_formatted_brackets(sentence: str, entity: tuple) -> bool:

    entity_text, start_pos, end_pos = entity

    text_before = sentence[:start_pos]
    text_after = sentence[end_pos:]

    last_open_bracket = text_before.rfind('(')
    first_close_bracket = text_after.find(')')

    if last_open_bracket != -1 and first_close_bracket != -1:
        bracket_content = sentence[last_open_bracket + 1:end_pos + first_close_bracket]

        format_patterns = [
            r'\d{1,2}:\d{2}\s+\w+/\d{1,2}:\d{2}\s+\w+',
            r'\d{4}\s+\w+/\d{4}\s+\w+',
            r'v?\d+\.\d+[\.\d]*\s*/\s*\w+',
            r'[\$€£¥]\d+/[\$€£¥]\d+',
            r'^[^/]{1,15}/[^/]{1,15}$'
        ]

        for pattern in format_patterns:
            if re.search(pattern, bracket_content):
                print(f"DEBUG - Skipping entity in formatted brackets: '{entity_text}' in '{bracket_content}'")
                return True

        if '/' in bracket_content and len(bracket_content.strip()) < 30:
            if re.match(r'^[\w\s/:.-]+$', bracket_content.strip()):
                print(f"DEBUG - Skipping entity in likely formatted brackets: '{entity_text}' in '{bracket_content}'")
                return True

    return False


def has_capitalized_word_after_entity(sentence: str, entity: tuple) -> bool:

    entity_text, start_pos, end_pos = entity

    # 
    text_after = sentence[end_pos:]

    if not text_after:
        return False

    import re

    pattern = r'^\s+([A-Z][a-z]+)'
    match = re.match(pattern, text_after)

    if match:
        capitalized_word = match.group(1)
        print(f"DEBUG - Found capitalized word immediately after entity: '{entity_text}' + '{capitalized_word}'")
        return True

    return False


def has_definite_article_before_entity(sentence: str, entity: tuple) -> bool:

    entity_text, start_pos, end_pos = entity

    text_before = sentence[:start_pos].rstrip()

    if not text_before:
        return False

    words_before = text_before.split()
    if words_before:
        last_word = words_before[-1].lower()
        if last_word == "the":
            print(f"DEBUG - Found definite article 'the' before entity '{entity_text}'")
            return True

    return False


def has_possessive_before_entity(sentence: str, entity: tuple) -> bool:

    entity_text, start_pos, end_pos = entity

    # 
    text_before = sentence[:start_pos].rstrip()

    if not text_before:
        return False

    possessive_patterns = [
        r"'s\s*$",  # 
        r"s'\s*$",  # 
        r"'\s*$",  # 
    ]

    import re
    for pattern in possessive_patterns:
        if re.search(pattern, text_before):
            print(f"DEBUG - Found possessive pattern before entity '{entity_text}'")
            return True

    if nlp:
        try:
            doc = nlp(sentence)

            # 
            entity_tokens = []
            for token in doc:
                if start_pos <= token.idx < end_pos:
                    entity_tokens.append(token)

            if entity_tokens:
                first_entity_token = entity_tokens[0]

                if first_entity_token.dep_ == "poss":
                    print(f"DEBUG - spaCy detected entity '{entity_text}' has possessive dependency")
                    return True

                if first_entity_token.i > 0:
                    prev_token = doc[first_entity_token.i - 1]
                    if prev_token.tag_ == "POS" or prev_token.text in ["'s", "'", "s'"]:
                        print(
                            f"DEBUG - spaCy detected possessive marker '{prev_token.text}' before entity '{entity_text}'")
                        return True

        except Exception as e:
            print(f"DEBUG - spaCy possessive analysis failed: {e}")

    return False


def has_number_before_entity(sentence: str, entity: tuple) -> bool:

    entity_text, start_pos, end_pos = entity

    # 
    text_before = sentence[:start_pos].strip()

    if not text_before:
        return False

    import re

    number_pattern = r'\b\d+(?:\.\d+)?(?:\s*%)?$'  # 

    if re.search(number_pattern, text_before):
        print(f"DEBUG - Found number+entity pattern: '{text_before}' + '{entity_text}', skipping")
        return True

    comma_number_pattern = r'\b\d{1,3}(?:,\d{3})+$'
    if re.search(comma_number_pattern, text_before):
        print(f"DEBUG - Found comma-separated number+entity pattern: '{text_before}' + '{entity_text}', skipping")
        return True

    quantifier_only_patterns = [
        r'\b(?:several|many|few|countless|innumerable|numerous)\s*$',
        r'\b(?:dozens?|hundreds?|thousands?|millions?)\s+of\s*$',
        r'\ba\s+(?:couple|few|dozen|hundred|thousand|million)\s+(?:of\s+)?$',
        r'\b(?:some|multiple)\s*$',
    ]

    for pattern in quantifier_only_patterns:
        if re.search(pattern, text_before, re.IGNORECASE):
            print(f"DEBUG - Found quantifier+entity pattern: '{text_before}' + '{entity_text}', skipping")
            return True

    if nlp:
        try:
            doc = nlp(text_before)

            tokens = [t for t in doc if not t.is_punct and not t.is_space]
            if tokens:
                for token in tokens[-2:]:
                    if token.pos_ == "NUM" or token.like_num:
                        print(f"DEBUG - spaCy detected number token '{token.text}' before entity '{entity_text}'")
                        return True

                    if token.lemma_.lower() in {"dozen", "hundred", "thousand", "million", "billion",
                                                "many", "few", "several", "some", "multiple"}:
                        print(f"DEBUG - spaCy detected quantifier '{token.text}' before entity '{entity_text}'")
                        return True

        except Exception as e:
            print(f"DEBUG - spaCy number analysis failed: {e}")

    return False


def has_ordinal_before_entity(sentence: str, entity: tuple) -> bool:

    entity_text, start_pos, end_pos = entity

    text_before = sentence[:start_pos].strip()

    if not text_before:
        return False

    import re

    ordinal_patterns = [
        r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)$',
        r'\b(?:eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)$',
        r'\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)$',
        r'\b\d+(?:st|nd|rd|th)$',  # 1st, 2nd, 3rd, 4th等
        r'\bthe\s+(?:first|second|third|last|final|initial|primary)$',
    ]

    for pattern in ordinal_patterns:
        if re.search(pattern, text_before, re.IGNORECASE):
            print(f"DEBUG - Found ordinal+entity pattern: '{text_before}' + '{entity_text}', skipping")
            return True

    return False


def has_wh_problem_before_entity_spacy(sentence: str, entity: tuple) -> bool:

    if nlp is None:
        print("DEBUG - spaCy not available for pronoun checking")
        return False

    entity_text, start_pos, end_pos = entity

    try:
        
        doc = nlp(sentence)

        entity_start_token = None
        for token in doc:
            if start_pos <= token.idx < end_pos:
                entity_start_token = token
                break

        if not entity_start_token or entity_start_token.i == 0:
            return False

        preceding_token = None
        for i in range(entity_start_token.i - 1, -1, -1):
            token = doc[i]
            if not token.is_punct and not token.is_space and token.text.strip():
                preceding_token = token
                break

        if not preceding_token:
            return False

        print(
            f"DEBUG - spaCy pronoun check: '{preceding_token.text}' (POS: {preceding_token.pos_}, TAG: {preceding_token.tag_}) + '{entity_text}'")

        if preceding_token.tag_ == "WP$":  # Wh-possessive
            print(f"DEBUG - spaCy: Found possessive wh-word '{preceding_token.text}' before '{entity_text}', skipping")
            return True

        if (preceding_token.text.lower() in ["these", "those"] and
                preceding_token.pos_ in ["DET", "PRON"]):
            print(f"DEBUG - spaCy: Found demonstrative '{preceding_token.text}' before '{entity_text}', skipping")
            return True

        if (preceding_token.text.lower() == "which" and
                preceding_token.tag_ in ["WDT", "WP"]):  # Wh-determiner or Wh-pronoun
            print(f"DEBUG - spaCy: Found 'which' before '{entity_text}', may need skipping")
            return True

        return False

    except Exception as e:
        print(f"DEBUG - spaCy pronoun analysis failed: {e}")
        return False


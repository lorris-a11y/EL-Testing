import re
import spacy

try:
    nlp = spacy.load('en_core_web_sm')
    print("DEBUG - spaCy model loaded for pronoun utils")
except:
    nlp = None
    print("Warning: spaCy model not available for pronoun utils")


def needs_possessive_pronoun(text_after_entity: str, entity_type: str) -> bool:

    text_stripped = text_after_entity.strip()
    if not text_stripped:
        print("DEBUG - Empty text after entity, no possessive needed")
        return False

    if re.match(r'^[^\w]*$', text_stripped):
        print(f"DEBUG - Only punctuation after entity: '{text_stripped}', no possessive needed")
        return False

    print(f"\nDEBUG - Possessive check for '{entity_type}': '{text_after_entity.strip()}'")

    if _should_never_use_possessive(entity_type, text_after_entity):
        return False

    if nlp is None:
        print("ERROR - spaCy not available, cannot perform possessive analysis")
        return False

    return _spacy_possessive_analysis(text_after_entity)


def _should_never_use_possessive(entity_type: str, text_after: str) -> bool:
    if entity_type in ["PERSON", "PER", "Person"]:
        print("DEBUG - Person entity, no possessive")
        return True

    if entity_type in {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "DateTime"}:
        print("DEBUG - Numeric entity, no possessive")
        return True

    if text_after.lstrip().startswith(','):
        print("DEBUG - Starts with comma, no possessive")
        return True

    return False


def _spacy_possessive_analysis(text_after: str) -> bool:
    doc = nlp(text_after)
    if not doc:
        print("DEBUG - Empty spaCy doc")
        return False

    first_token = doc[0] if len(doc) > 0 else None
    if first_token and first_token.lemma_ in ["be", "become", "seem", "appear", "look", "sound", "smell", "taste",
                                              "feel", "remain", "stay"]:
        print(f"DEBUG - Found copula verb '{first_token.text}', returning False")
        return False

    if _is_geographic_location_pattern(doc):
        print("DEBUG - Geographic location pattern detected")
        return False

    first_meaningful_token = _get_first_meaningful_token(doc)
    if not first_meaningful_token:
        print("DEBUG - No meaningful token found")
        return False

    print(
        f"DEBUG - First meaningful token: '{first_meaningful_token.text}' (POS: {first_meaningful_token.pos_}, DEP: {first_meaningful_token.dep_})")

    if _is_non_possessive_pos(first_meaningful_token):
        return False

    if first_meaningful_token.pos_ in ["NOUN", "PROPN"]:
        if _is_part_of_named_entity(first_meaningful_token, doc):
            print("DEBUG - Noun is part of named entity")
            return False

        print("DEBUG - Found standalone noun, needs possessive")
        return True

    if _has_compound_noun_pattern(doc):
        print("DEBUG - Compound noun pattern detected")
        return True

    print("DEBUG - spaCy analysis inconclusive, using rule-based final check")
    return _final_content_word_check(text_after)


def _get_first_meaningful_token(doc):
    for token in doc:
        if not token.is_punct and not token.is_space and token.text.strip():
            return token
    return None


def _is_geographic_location_pattern(doc):
    if len(doc.ents) > 0 and doc.text.lstrip().startswith(','):
        location_entities = [ent for ent in doc.ents[:2] if ent.label_ in ["GPE", "LOC", "FAC"]]
        return len(location_entities) > 0
    return False


def _is_non_possessive_pos(token):
    if token.lemma_ in ["be", "become", "seem", "appear", "look", "sound", "smell", "taste", "feel", "remain", "stay"]:
        print(f"DEBUG - Found copula verb '{token.text}'")
        return True

    if token.pos_ == "ADP":
        print(f"DEBUG - Found preposition '{token.text}'")
        return True

    if token.pos_ == "DET":
        print(f"DEBUG - Found determiner '{token.text}'")
        return True

    if token.pos_ in ["VERB", "AUX", "CCONJ", "SCONJ"]:
        print(f"DEBUG - Found {token.pos_} '{token.text}'")
        return True

    if token.pos_ == "PART":
        print(f"DEBUG - Found particle/infinitive marker '{token.text}'")
        return True

    if token.pos_ == "NUM":
        print(f"DEBUG - Found number '{token.text}'")
        return True

    if token.pos_ == "ADV":
        print(f"DEBUG - Found adverb '{token.text}'")
        return True

    if token.text.lower() in ["which", "that", "who", "whom", "whose"]:
        print(f"DEBUG - Found relative pronoun '{token.text}'")
        return True

    return False


def _is_part_of_named_entity(token, doc):
    return any(token.i >= ent.start and token.i < ent.end for ent in doc.ents)


def _has_compound_noun_pattern(doc):
    meaningful_tokens = [t for t in doc if not t.is_punct and not t.is_space]

    for i in range(min(3, len(meaningful_tokens) - 1)):
        curr_token = meaningful_tokens[i]
        next_token = meaningful_tokens[i + 1]

        if (curr_token.pos_ in ["NOUN", "PROPN"] and
                next_token.pos_ in ["NOUN", "PROPN"]):

            between_tokens = doc[curr_token.i:next_token.i]
            has_preposition = any(t.pos_ == "ADP" for t in between_tokens)

            if not has_preposition:
                print("DEBUG - Found compound noun pattern")
                return True

    return False


def _final_content_word_check(text: str) -> bool:
    from .constants import FUNCTION_WORDS

    text = text.lstrip()

    function_word_pattern = '|'.join(re.escape(word) for word in FUNCTION_WORDS)
    function_words_regex = f'^(?:{function_word_pattern})\\s+'

    if re.match(function_words_regex, text, re.IGNORECASE):
        first_word_match = re.match(r'^(\S+\s+)', text)
        if first_word_match:
            text = text[first_word_match.end():]

    content_words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not content_words:
        print("DEBUG - No content words found")
        return False

    first_word = content_words[0].lower()

    if first_word in FUNCTION_WORDS:
        print(f"DEBUG - Found non-possessive word: {first_word}")
        return False

    print(f"DEBUG - Found possessive-worthy content word: {first_word}")
    return True


from .constants import FEMALE_INDICATORS, MALE_INDICATORS


def detect_gender_in_sentence(sentence: str) -> str:
    sentence_lower = sentence.lower()

    female_count = sum(1 for word in FEMALE_INDICATORS if f' {word} ' in f' {sentence_lower} ')
    male_count = sum(1 for word in MALE_INDICATORS if f' {word} ' in f' {sentence_lower} ')

    print(f"DEBUG - Gender detection: female_count={female_count}, male_count={male_count}")

    if female_count > male_count:
        return 'female'
    elif male_count > female_count:
        return 'male'
    else:
        return 'unknown'



def select_pronoun_universal(
        entity_type: str,
        context_after: str = "",
        sentence: str = "",
        context_before: str = "",
        ner_model: str = "conll3"
) -> str:

    from .constants import PREPOSITIONS

    print(f"\nDEBUG - Selecting pronoun:")
    print(f"Entity type: {entity_type}")
    print(f"NER model: {ner_model}")
    print(f"Context before: '{context_before}'")
    print(f"Context after: '{context_after}'")

    std_type = _standardize_entity_type_for_pronoun(entity_type, ner_model)

    if context_before:
        words = context_before.strip().split()
        if words:
            last_word = words[-1].lower().rstrip('.,!?;:')
            if last_word in PREPOSITIONS:
                print(f"DEBUG - Found preposition '{last_word}', using objective case")
                return _get_objective_pronoun(std_type, sentence)

    needs_possessive = needs_possessive_pronoun(context_after, std_type)
    print(f"DEBUG - Needs possessive: {needs_possessive}")

    if needs_possessive:
        return _get_possessive_pronoun(std_type, sentence)

    return _get_nominative_pronoun(std_type, sentence)


def _standardize_entity_type_for_pronoun(entity_type: str, ner_model: str) -> str:
    ner_model_lower = ner_model.lower()

    person_types = {
        "conll3": ["PER"],
        "ontonotes": ["PERSON"],
        "azure": ["Person"],
        "aws": ["PERSON"]
    }

    group_types = {
        "conll3": [],
        "ontonotes": ["NORP"],
        "azure": ["PersonType"],
        "aws": []
    }

    if ner_model_lower in person_types and entity_type in person_types[ner_model_lower]:
        return "PERSON"

    if ner_model_lower in group_types and entity_type in group_types[ner_model_lower]:
        return "GROUP"

    return "OTHER"


def _get_nominative_pronoun(std_type: str, sentence: str = "") -> str:
    if std_type == "PERSON":
        if sentence:
            detected_gender = detect_gender_in_sentence(sentence)
            print(f"DEBUG - Detected gender: {detected_gender}")
            return "she" if detected_gender == 'female' else "he"
        return "he"
    elif std_type == "GROUP":
        return "they"
    else:
        return "it"


def _get_objective_pronoun(std_type: str, sentence: str = "") -> str:
    if std_type == "PERSON":
        if sentence:
            detected_gender = detect_gender_in_sentence(sentence)
            print(f"DEBUG - Detected gender: {detected_gender}")
            return "her" if detected_gender == 'female' else "him"
        return "him"
    elif std_type == "GROUP":
        return "them"
    else:
        return "it"


def _get_possessive_pronoun(std_type: str, sentence: str = "") -> str:
    if std_type == "PERSON":
        if sentence:
            detected_gender = detect_gender_in_sentence(sentence)
            return "her" if detected_gender == 'female' else "his"
        return "his"
    elif std_type == "GROUP":
        return "their"
    else:
        return "its"
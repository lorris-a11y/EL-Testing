
import re
from dataclasses import dataclass
from typing import Dict, List

import spacy
from flair.data import Sentence

from external.plural import (
    is_verb_plural, is_verb_singular, pluralize_verb, singularize_verb,
    is_word_plural
)
from external.plural.addition import is_passive_structure
from external.sentence import MutationSentence
from external.syntax import recover_word
from external.tense.detection import is_simple_past, is_past_participle

# Load spaCy model
nlp = spacy.load('en_core_web_sm')


@dataclass
class WhoClause:
    antecedent: spacy.tokens.Token
    verb: spacy.tokens.Token
    is_plural: bool


@dataclass
class NumericPatternManager:

    PATTERNS = [
        #
        (r'\([^)]*\d+(?:,\d{3})*[^)]*\)', 'parenthesized'),

        #
        (r'[£€$]\d+(?:,\d{3})*(?:\.\d+)?', 'currency'),

        #  -
        (r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', 'number')
    ]

    def __init__(self, sentence: str):
        self.sentence = sentence
        self.protected_map: Dict[str, str] = {}
        self.protected_sentence = sentence

    def protect_patterns(self) -> str:
        """"""
        self.protected_sentence = re.sub(r'(\d),\s+(\d)', r'\1,\2', self.protected_sentence)

        # /
        for pattern, pattern_type in self.PATTERNS:
            matches = list(re.finditer(pattern, self.protected_sentence))
            for match in reversed(matches):
                full_text = match.group(0)

                #
                if any(full_text == v for v in self.protected_map.values()):
                    continue

                #
                placeholder = f"__NUM_{len(self.protected_map)}__"
                self.protected_map[placeholder] = full_text

                #
                self.protected_sentence = (
                        self.protected_sentence[:match.start()] +
                        placeholder +
                        self.protected_sentence[match.end():]
                )

        return self.protected_sentence

    def restore_patterns(self, text: str) -> str:
        """"""
        result = text

        #
        placeholders = sorted(self.protected_map.keys(), key=len, reverse=True)

        for placeholder in placeholders:
            original = self.protected_map[placeholder]

            #
            if placeholder in result:
                result = result.replace(placeholder, original)
                continue

            # （）
            base_num = re.search(r'\d+', placeholder)
            if base_num:
                num = base_num.group(0)
                #
                corrupted_pattern = fr'__[^_]*{num}[^_]*__'
                for match in re.finditer(corrupted_pattern, result):
                    result = result.replace(match.group(0), original)

        return result.strip()


def clean_punctuation(text: str) -> str:
    """Clean up punctuation while preserving number formatting"""
    # First protect numbers from space normalization
    number_pattern = r'\b\d{1,3}(?:,\d{3})*\b'
    numbers = {}
    for i, match in enumerate(re.finditer(number_pattern, text)):
        placeholder = f'__NUM_{i}__'
        numbers[placeholder] = match.group(0)
        text = text[:match.start()] + placeholder + text[match.end:]

    # Clean up spaces and punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\.\s*$', '.', text)
    text = re.sub(r'\s*,\s*and\s*', ' and ', text)
    text = re.sub(r'\s*,\s*or\s*', ' or ', text)

    # Restore numbers
    for placeholder, number in numbers.items():
        text = text.replace(placeholder, number)

    return text.strip()

#
# """"""
def find_who_clauses(doc: spacy.tokens.Doc) -> List[WhoClause]:
    who_clauses = []
    for token in doc:
        if token.text.lower() != 'who':
            continue

        antecedent = next((ancestor for ancestor in token.ancestors
                           if ancestor.dep_ in ["nsubj", "nsubjpass"]), None)

        clause_verb = next((descendant for descendant in token.subtree
                            if descendant.dep_ == "relcl" and descendant.pos_ == "VERB"), None)

        if antecedent and clause_verb:
            who_clauses.append(WhoClause(
                antecedent=antecedent,
                verb=clause_verb,
                is_plural=is_word_plural(antecedent.text)
            ))

    return who_clauses
#
#
""""""
def should_keep_conjunction(token: spacy.tokens.Token) -> bool:
    if token.text.lower() not in ['and', 'or']:
        return True

    #
    if any(ent.label_ in ['GPE', 'LOC'] for ent in token.doc.ents):
        return True

    #
    prev_text = token.doc[token.i - 2:token.i].text if token.i >= 2 else ""
    next_text = token.doc[token.i + 1:token.i + 3].text if token.i < len(token.doc) - 2 else ""
    if re.search(r'\b(?:part of|area|region)\b', prev_text):
        return True

    #
    if token.text.lower() == 'and':
        #  between
        prev_text = ''.join(t.text_with_ws for t in token.doc[:token.i]).lower()
        if 'between' in prev_text:
            return True

    prev_token = token.doc[token.i - 1] if token.i > 0 else None
    next_token = token.doc[token.i + 1] if token.i < len(token.doc) - 1 else None

    prev_entity = next((token.doc[j] for j in range(token.i - 1, -1, -1)
                        if token.doc[j].ent_type_), None)
    next_entity = next((token.doc[j] for j in range(token.i + 1, len(token.doc))
                        if token.doc[j].ent_type_), None)

    return ((prev_entity and next_entity) or
            (prev_token and next_token and (
                    prev_token.text == ',' or
                    (prev_token.pos_ == next_token.pos_ and
                     not prev_token.is_punct and
                     not next_token.is_punct)
            )))



# #new
def restore_special_forms(text: str, original_sentence: str) -> str:
    #
    contractions = re.findall(r"\w+n't|\w+'[sd]|\w+'ll|\w+'ve|\w+'re", original_sentence)
    for contraction in contractions:
        parts = contraction.split("'")
        if len(parts) > 1:
            text = text.replace(f"{parts[0]} '{parts[1]}", contraction)
        if "n't" in contraction:
            text = text.replace(contraction.replace("n't", " n't"), contraction)
            text = text.replace(
                contraction.capitalize().replace("n't", " n't"),
                contraction.capitalize()
            )

    # （）
    possessive_patterns = [r"\b(\w+)(\s+)['']s\b", r"\b(\w+\s+\w+)(\s+)['']s\b"]
    for pattern in possessive_patterns:
        for match in re.finditer(pattern, text):
            original = match.group(0)
            fixed = match.group(1) + "'" + match.group(0)[-1]  #
            text = text.replace(original, fixed)


    multi_word_names = re.findall(r'[A-Z][a-z]+\s+[A-Z][a-z]+', original_sentence)
    for name in multi_word_names:
        parts = name.split()
        if len(parts) == 2:
            #
            joined_name = ''.join(parts)
            if joined_name in text:
                text = text.replace(joined_name, name)

    #
    hyphenated_words = re.findall(r'\b\w+(?:-\w+)+\b', original_sentence)
    for word in hyphenated_words:
        wrong_forms = [
            ' '.join(word.split('-')),
            '-'.join(' '.join(word.split('-')).split()),
            ' - '.join(word.split('-')),
            '- '.join(word.split('-')),
            ' -'.join(word.split('-'))
        ]
        for wrong_form in wrong_forms:
            text = text.replace(wrong_form, word)

    return text










def refine_sentence_with_spacy(sentence: str, tagger) -> str:

    # Step 1:
    pattern_manager = NumericPatternManager(sentence)
    protected_sentence = pattern_manager.protect_patterns()

    # Step 2:
    doc = nlp(protected_sentence)
    flair_sentence = Sentence(sentence)
    tagger.predict(flair_sentence)

    #
    original_tokens = []
    original_pos = []
    original_deps = []
    original_head = []

    for token in doc:
        original_tokens.append({
            'text': token.text,
            'idx': token.i,
            'starts_at': token.idx,
            'ends_at': token.idx + len(token.text),
            'whitespace_after': token.whitespace_,
            'is_contraction_part': token.text.lower() in ["'s", "'ve", "'re", "'d", "'ll"]
                                   or token.text == "n't"
        })
        original_pos.append(token.pos_)
        original_deps.append(token.dep_)
        original_head.append(token.head)

    #
    mutation_sentence = MutationSentence([token['text'] for token in original_tokens])

    #
    noun_indices = []
    for i, token in enumerate(doc):
        if (token.pos_ in ["NOUN", "PROPN"]) and token.dep_ in ["nsubj", "nsubjpass"]:
            noun_indices.append(i)

    #
    for noun_index in noun_indices:
        token = mutation_sentence[noun_index]
        verb = original_head[noun_index]
        verb_index = verb.i

        #
        if is_passive_structure(doc, verb_index):
            #
            aux_index = -1
            for i, dep in enumerate(original_deps):
                if dep in ["aux", "auxpass"] and original_head[i].i == verb_index:
                    aux_index = i
                    #
                    break

            if aux_index != -1:
                aux_token = mutation_sentence[aux_index]
                # ，
                if not is_word_plural(token) and aux_token.lower() == "have":
                    mutation_sentence[aux_index] = recover_word("has", aux_token)
                #
                elif is_word_plural(token) and aux_token.lower() == "has":
                    mutation_sentence[aux_index] = recover_word("have", aux_token)

            #
            continue

        #
        #
        aux_index = -1
        for i, dep in enumerate(original_deps):
            if dep == "aux" and original_head[i].i == verb_index:
                aux_index = i
                break

        if aux_index != -1:
            aux_token = mutation_sentence[aux_index]
            #
            if not is_word_plural(token) and aux_token.lower() == "have":
                mutation_sentence[aux_index] = recover_word("has", aux_token)
            #
            elif is_word_plural(token) and aux_token.lower() == "has":
                mutation_sentence[aux_index] = recover_word("have", aux_token)
        else:
            #
            verb_token = mutation_sentence[verb_index]

            #
            if is_past_participle(verb_token) or is_simple_past(verb_token):
                continue

            # ，
            if not is_word_plural(token) and is_verb_plural(verb_token):
                mutation_sentence[verb_index] = recover_word(
                    singularize_verb(verb_token),
                    verb_token
                )
            #
            elif is_word_plural(token) and is_verb_singular(verb_token):
                mutation_sentence[verb_index] = recover_word(
                    pluralize_verb(verb_token),
                    verb_token
                )

    #
    who_clauses = find_who_clauses(doc)
    for clause in who_clauses:
        current_verb = mutation_sentence[clause.verb.i]

        #
        if is_past_participle(current_verb) or is_simple_past(current_verb):
            continue

        if clause.is_plural and is_verb_singular(current_verb):
            mutation_sentence[clause.verb.i] = recover_word(
                pluralize_verb(current_verb), current_verb
            )
        elif not clause.is_plural and is_verb_plural(current_verb):
            mutation_sentence[clause.verb.i] = recover_word(
                singularize_verb(current_verb), current_verb
            )

    #
    result = ""
    for i, token_info in enumerate(original_tokens):
        current_token = mutation_sentence[i]

        #
        if i > 0:
            prev_token = original_tokens[i - 1]
            # ，
            if token_info['is_contraction_part']:
                pass
            # ，
            elif prev_token['whitespace_after']:
                result += ' '

        #
        result += current_token

        if i == len(original_tokens) - 1 and token_info['whitespace_after']:
            result += ' '

    #
    result = restore_special_forms(result, sentence)
    result = clean_punctuation(result)

    #
    result = pattern_manager.restore_patterns(result)

    return result.strip()


def refine_sentence(sentence: str, tagger) -> str:
    return refine_sentence_with_spacy(sentence, tagger)


'''
plural and singular
'''
from typing import List
import spacy
import nltk

from .addition import is_past_participle, is_simple_past
from ..sentence import MutationSentence
from ..syntax import find_child_index, recover_word

en_nlp = spacy.load('en_core_web_sm', disable=['parser', 'lemmatizer', 'custom'])
lemma = nltk.wordnet.WordNetLemmatizer()

from .irregular import PLURAL_SINGULAR_DICT, SINGULAR_PLURAL_DICT, VERB_AUX_PLURAL_SINGULAR, VERB_AUX_SINGULAR_PLURAL

def singularize(word: str) -> str:
    origin_word = word
    word = origin_word.strip().lower()

    ret = ""
    if word in PLURAL_SINGULAR_DICT:
        ret = PLURAL_SINGULAR_DICT[word]
    else:
        lemma_word = lemma.lemmatize(word, "n")
        ret = lemma_word
    
    if origin_word[0].isupper():
        ret = ret.capitalize()
    if origin_word.endswith(" "):
        ret = ret + " "
    
    return ret

def pluralize(word: str) -> str:
    origin_word = word
    word = origin_word.strip().lower()

    ret = ""
    if word in SINGULAR_PLURAL_DICT:
        ret = SINGULAR_PLURAL_DICT[word]
    elif word.endswith("s") or word.endswith("x") or word.endswith("z"):
        ret = word + "es"
    elif word.endswith("ch") or word.endswith("sh"):
        ret = word + "es"
    elif word.endswith("y"):
        ret = word[:-1] + "ies"
    else:
        ret = word + "s"
    
    if origin_word[0].isupper():
        ret = ret.capitalize()
    if origin_word.endswith(" "):
        ret = ret + " "
    
    return ret


def singularize_verb(word: str) -> str:
    origin_word = word
    word = origin_word.strip().lower()

    ret = ""
    if word in VERB_AUX_PLURAL_SINGULAR:
        ret = VERB_AUX_PLURAL_SINGULAR[word]
    elif word.endswith("s") or word.endswith("x") or word.endswith("z"):
        ret = word + "es"
    elif word.endswith("ch") or word.endswith("sh"):
        ret = word + "es"
    elif word.endswith("y"):
        ret = word[:-1] + "ies"
    else:
        ret = word + "s"
    
    if origin_word[0].isupper():
        ret = ret.capitalize()
    if origin_word.endswith(" "):
        ret = ret + " "
    
    return ret

def pluralize_verb(word: str) -> str:
    origin_word = word
    word = origin_word.strip().lower()

    lemma_word = lemma.lemmatize(word, "v")

    ret = ""
    if word in VERB_AUX_SINGULAR_PLURAL:
        ret = VERB_AUX_SINGULAR_PLURAL[word]
    else:
        ret = lemma_word
    
    if origin_word[0].isupper():
        ret = ret.capitalize()
    if origin_word.endswith(" "):
        ret = ret + " "
    
    return ret

def is_word_plural(word: str) -> bool:
    if word in PLURAL_SINGULAR_DICT or word in VERB_AUX_SINGULAR_PLURAL:
        return True
    return word.endswith("s") or word.endswith("es")

def is_word_singular(word: str) -> bool:
    return not is_word_plural(word)

# def is_verb_singular(word: str) -> bool:
#     if word in VERB_AUX_SINGULAR_PLURAL:
#         return True
#     return word.endswith("s") or word.endswith("es")
#
# def is_verb_plural(word: str) -> bool:
#     return not is_verb_singular(word)


# # 修改is_verb_plural和is_verb_singular函数，避免将过去分词识别为复数
# def is_verb_singular(word: str, check_participle: bool = True) -> bool:
#     """判断动词是否是单数形式，增加过去分词检查"""
#     # 先检查是否是过去分词，如果是则返回False（既不是单数也不是复数）
#     if check_participle and is_past_participle(word.strip().lower()):
#         return False
#
#     if word.strip().lower() in VERB_AUX_SINGULAR_PLURAL:
#         return True
#
#     return word.strip().lower().endswith("s") or word.strip().lower().endswith("es")

#def is_verb_plural(word: str, check_participle: bool = True) -> bool:
#     """判断动词是否是复数形式，增加过去分词检查"""
#     # 先检查是否是过去分词，如果是则返回False（既不是单数也不是复数）
#     if check_participle and is_past_participle(word.strip().lower()):
#         return False
#
#     # 非过去分词且非单数则为复数
#     return not is_verb_singular(word, check_participle=False)

#new version
def is_verb_singular(word: str, check_tense: bool = True) -> bool:
    """判断动词是否是单数形式，增加时态检查"""
    word = word.strip().lower()

    # 先检查是否是过去分词或过去式，如果是则返回False（既不是单数也不是复数）
    if check_tense:
        # 检查是否是过去分词
        if is_past_participle(word):
            return False

        # 检查是否是简单过去式
        if is_simple_past(word):
            return False

        # 检查是否是通用的过去式（大多数规则动词以ed结尾）
        if word.endswith('ed') and not word in ['need', 'feed', 'seed', 'deed', 'speed', 'breed']:
            return False

    # 检查是否是已知的单数动词形式
    if word in VERB_AUX_SINGULAR_PLURAL:
        return True

    # 检查语法规则
    return word.endswith("s") or word.endswith("es") or word.endswith("ies")


def is_verb_plural(word: str, check_tense: bool = True) -> bool:
    """判断动词是否是复数形式，增加时态检查"""
    word = word.strip().lower()

    # 先检查是否是过去分词或过去式，如果是则返回False（既不是单数也不是复数）
    if check_tense:
        # 检查是否是过去分词
        if is_past_participle(word):
            return False

        # 检查是否是简单过去式
        if is_simple_past(word):
            return False

        # 检查是否是通用的过去式（大多数规则动词以ed结尾）
        if word.endswith('ed') and not word in ['need', 'feed', 'seed', 'deed', 'speed', 'breed']:
            return False

    # 检查是否是已知的复数动词形式
    if word in VERB_AUX_PLURAL_SINGULAR:
        return True

    # 检查语法规则 - 非单数即为复数
    return not is_verb_singular(word, check_tense=False)




def is_singular(original_tokens: List[str], original_pos: List[str], original_entities: List[spacy.tokens.Span], i: int) -> bool:
    if i == 0:
        return False
    token = original_tokens[i].strip().lower()
    prev_token = original_tokens[i - 1].strip().lower()
    pos = original_pos[i]
    if original_pos[i] == "NOUN":
        if prev_token in ["a", "an"]:
            return True
        return False     
    else:
        return False

def is_plural(original_tokens: List[str], original_pos: List[str], original_entities: List[spacy.tokens.Span], i: int) -> bool:
    if i == 0:
        return False
    token = original_tokens[i].strip().lower()
    prev_token = original_tokens[i - 1].strip().lower()
    pos = original_pos[i]
    if original_pos[i] == "NOUN":
        if prev_token in ["some", "many"] and is_word_plural(token):
            return True
        return False
    else:
        return False

# 添加函数识别过去分词

def is_past_participle(word: str) -> bool:
    """判断一个词是否是过去分词"""
    # 常见的规则过去分词以ed结尾
    if word.lower().endswith('ed'):
        return True

    # 常见的不规则过去分词列表
    irregular_participles = {
        'known', 'done', 'gone', 'given', 'taken', 'seen', 'written',
        'spoken', 'broken', 'chosen', 'forgotten', 'frozen', 'stolen',
        'worn', 'torn', 'born', 'built', 'bought', 'brought', 'caught',
        'found', 'heard', 'held', 'kept', 'made', 'meant', 'met', 'paid',
        'read', 'said', 'sold', 'sent', 'shown', 'shut', 'sung', 'sat',
        'slept', 'spent', 'taught', 'told', 'thought', 'understood', 'won'
    }

    return word.lower() in irregular_participles


def is_simple_past(word: str) -> bool:
    """判断一个词是否是简单过去式"""
    irregular_past = {
        'ran', 'went', 'saw', 'ate', 'came', 'took', 'gave', 'knew',
        'grew', 'drew', 'threw', 'spoke', 'drove', 'wrote', 'rode',
        'rose', 'broke', 'chose', 'froze', 'got', 'sat', 'led',
        'left', 'fought', 'felt', 'fell', 'found', 'flew', 'lost',
        'told', 'held', 'heard', 'let', 'read', 'meant', 'sent',
        'built', 'bought', 'brought', 'caught', 'cost', 'cut', 'hit',
        'hurt', 'put', 'set', 'shot', 'shut', 'spent', 'split',
        'stood', 'thought', 'understood', 'won'
    }

    # 大多数规则动词的过去式以ed结尾，但这与过去分词重叠
    # 因此主要依赖不规则动词列表
    return word.lower() in irregular_past

def is_passive_structure(doc, verb_index: int) -> bool:
    """判断是否是被动结构 (is/was/are/were + past participle)"""
    verb = doc[verb_index]

    # 检查是否有aux或auxpass依存关系的辅助动词
    aux_verbs = [token for token in doc if token.dep_ in ['aux', 'auxpass'] and token.head.i == verb_index]

    if not aux_verbs:
        return False

    # 检查辅助动词是否是be动词的某种形式
    be_forms = ['is', 'am', 'are', 'was', 'were', 'be', 'been', 'being']
    if any(aux.lemma_ == 'be' or aux.text.lower() in be_forms for aux in aux_verbs):
        # 检查主动词是否是过去分词
        return is_past_participle(verb.text)

    return False

HONORIFICS = [
    'Mr', 'Mr.', 'Mrs', 'Mrs.', 'Ms', 'Ms.',
    'Dr', 'Dr.', 'Prof', 'Prof.', 'Professor',
    'Sir', 'Dame', 'Lord', 'Lady'
]

HONORIFICS_SET = {h.lower().rstrip('.') for h in HONORIFICS}



POSITION_TITLES = {

    'governor', 'president', 'vice president', 'senator', 'congressman', 'congresswoman',
    'mayor', 'minister', 'secretary', 'ambassador', 'judge', 'justice',


    'ceo', 'chief executive', 'chief executive officer', 'cfo', 'chief financial officer',
    'cto', 'chief technology officer', 'coo', 'chief operating officer', 'chairman',
    'chairwoman', 'director', 'manager', 'supervisor', 'coordinator',


    'professor', 'lecturer', 'researcher', 'scientist', 'scholar', 'dean',


    'general', 'colonel', 'captain', 'lieutenant', 'sergeant', 'admiral',


    'doctor', 'physician', 'surgeon', 'nurse', 'therapist',


    'lawyer', 'attorney', 'counsel', 'advocate', 'solicitor', 'barrister',


    'pastor', 'priest', 'bishop', 'rabbi', 'imam', 'minister',


    'engineer', 'architect', 'consultant', 'analyst', 'specialist', 'expert',
    'chief', 'head', 'leader', 'commander', 'supervisor'
}



GENERIC_ENTITIES = {
    'person', 'people', 'organization', 'company', 'location', 'place',
    'product', 'thing', 'entity', 'item'
}



PREPOSITIONS = {
    'at', 'in', 'on', 'by', 'with', 'from', 'to', 'for', 'of', 'through',
    'across', 'under', 'over', 'above', 'below', 'between', 'among',
    'during', 'before', 'after', 'since', 'until', 'within', 'without'
}


COPULA_VERBS = {
    'be', 'become', 'seem', 'appear', 'look', 'sound', 'smell', 'taste',
    'feel', 'remain', 'stay'
}


FUNCTION_WORDS = {

    'in', 'on', 'at', 'by', 'with', 'from', 'to', 'for', 'of', 'through',

    'the', 'a', 'an',

    'and', 'or', 'but', 'if', 'that', 'which', 'when', 'where', 'who', 'whom',

    'he', 'she', 'it', 'they', 'we', 'you', 'i',

    'is', 'was', 'are', 'were', 'am', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
    'can', 'may', 'might', 'must', 'shall', 'ought'
}


RELATIVE_PRONOUNS = {'which', 'that', 'who', 'whom', 'whose'}



#
FEMALE_INDICATORS = [
    'she', 'her', 'hers', 'mrs', 'mrs.', 'ms', 'ms.', 'miss',
    'woman', 'girl', 'lady', 'mother', 'daughter', 'sister', 'wife'
]

MALE_INDICATORS = [
    'he', 'him', 'his', 'mr', 'mr.', 'sir',
    'man', 'boy', 'gentleman', 'father', 'son', 'brother', 'husband'
]
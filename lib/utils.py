TYPES_MAP = {
    'EQUI': 7,
    'OPPO': 6,
    'SPE1': 5,
    'SPE2': 4,
    'SIMI': 3,
    'REL': 2,
    'ALIC': 1,
    'NOALI': 0,
}

def types_to_int(types):
    return list(map(lambda x: TYPES_MAP[x], types))

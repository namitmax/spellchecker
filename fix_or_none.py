def fix_or_none_bigram(clear_language_model, word, previous, pos, a=1, b=1, cond=1e-6):
    first = previous if pos == 0 else word
    second = word if pos == 0 else previous 
    if clear_language_model.GetWeight(first, second) > cond:
        return True
    return False

def get_nice_bad_els_bigram(inp, clear_language_model):
    query_length = len(inp)
    again = False
    was_bad = True
    pos = 1
    bad_els = []
    nice_els = []
    for i, word in enumerate(inp):
        if again:
            again = False
            pos = 0
            continue
        if i < query_length - 1 and was_bad:
            previous = inp[i + 1]
            pos = 1
        if i < query_length - 1 and fix_or_none_bigram(clear_language_model, word, previous, pos, 1, 1):
            if not was_bad:
                nice_els.append(i)
                previous = word
                pos = 0
            else:
                again = True
                nice_els.append(i)
                nice_els.append(i + 1)
            was_bad = False
            continue
        bad_els.append(i)
    return nice_els, bad_els

def fix_or_none_uni(clear_language_model, word, cond=1e-10):
    if clear_language_model.GetWeight(word, '') > cond:
        return True
    return False

def get_nice_bad_els(inp, clear_language_model, check_bigram):
    bad_els = []
    nice_els = []
    if check_bigram:
        return get_nice_bad_els_bigram(inp, clear_language_model)
    for i, word in enumerate(inp):
        if fix_or_none_uni(clear_language_model, word):
            nice_els.append(i)
            continue
        bad_els.append(i)
    return nice_els, bad_els
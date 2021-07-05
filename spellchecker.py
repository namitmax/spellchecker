import pickle
import numpy as np
import sys
import re
from string import punctuation
from Trie import Trie
from utils import get_query_proba
from utils import generate_candidates
from split_join_keyboard import Keyboard
from split_join_keyboard import SplitFixes
from split_join_keyboard import JoinFixes
from find_query import get_best
from find_query import get_best_queries
from find_query import FindBestQuery
from fix_or_none import get_nice_bad_els
from dataclasses import dataclass, field
from typing import Any
import copy

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)

def count_unigrams_w(query, clear_language_model):
    count = 0
    for word in (query):
        count += clear_language_model.GetWeight(word, '')
    return count

def count_bigrams_w(query, clear_language_model):
    count = 0
    for i, word in enumerate(query):
        if query[-1] == word:
            break
        count += clear_language_model.GetWeight(word, query[i+1])
    return count        

def count_unigrams(query, clear_language_model):
    count = 0
    for word in (query):
        if clear_language_model.GetWeight(word, '') > 1e-10:
            count += 1
    return count

def count_bigrams(query, clear_language_model):
    count = 0
    for i, word in enumerate(query):
        if query[-1] == word:
            break
        if clear_language_model.GetWeight(word, query[i+1]) > 1e-10:
            count += 1
    return count

def get_query(query):
    inp = []
    pre_tokens = re.sub('[' + punctuation + ']', '', query).split()
    return pre_tokens

def get_fix(inp, trie, keyboard, split, join, language_model, clear_language_model):
    inp = get_query(inp)
    l = len(inp)
    bigrams_count_inp = count_bigrams_w(inp, language_model)
    unigrams_count_inp = count_unigrams_w(inp, language_model)
    #print(bigrams_count_inp, unigrams_count_inp)
    keyboard_fixes = [keyboard.change_keyboard(word) for word in inp]
    bigrams_count_keyboard = count_bigrams_w(keyboard_fixes, language_model)
    unigrams_count_keyboard = count_unigrams_w(keyboard_fixes, language_model)
    #print(bigrams_count_keyboard, unigrams_count_keyboard)
    if (bigrams_count_keyboard > bigrams_count_inp):
        inp = keyboard_fixes
        bigrams_count_inp = bigrams_count_keyboard
        unigrams_count_inp = unigrams_count_keyboard
    elif (bigrams_count_keyboard == bigrams_count_inp and \
        unigrams_count_keyboard > unigrams_count_inp):
        inp = keyboard_fixes
        unigrams_count_inp = unigrams_count_keyboard
        
    bigrams_count_inp = count_bigrams(inp, language_model)
    unigrams_count_inp = count_unigrams(inp, language_model)
    #print(inp)
    join_fixes = join.fix_join(inp)
    inp = split.fix_split(inp)
    #print('1', inp)
    #print('2', join_fixes)
    epoch = 0
    fixes = copy.copy(inp)
    while (epoch < 2):
        bigrams_count_inp = count_bigrams(fixes, language_model)
        unigrams_count_inp = count_unigrams(fixes, language_model)
        check_bigram = False if epoch == 0 else True
        nice_els, bad_els = get_nice_bad_els(fixes, clear_language_model, check_bigram)
        if (len(bad_els) == 0 or (len(nice_els) < l and epoch == 1)):
            epoch += 1
            continue
        candidates, bad_els, nice_els = generate_candidates(trie, fixes, language_model, \
                                                clear_language_model, nice_els, bad_els)
        founder = FindBestQuery(fixes, candidates, language_model, \
                                clear_language_model, nice_els, bad_els)
        founder.get_ord_fixes()
        is_bad = (len(nice_els) == 0)
        fixes_results = get_best_queries(founder, nice_els, 5, is_bad)
        temp_result = get_best(fixes_results, language_model, clear_language_model)
        bigrams_count_fix = count_bigrams(temp_result, language_model)
        unigrams_count_fix = count_unigrams(temp_result, language_model)
        if (unigrams_count_fix < unigrams_count_inp or \
           bigrams_count_fix < bigrams_count_inp):
            epoch += 1
            continue
        fixes = temp_result
        epoch += 1
    #############################
    results = ['', '']
    #print(len(join_fixes), join_fixes)
    tmp = len(join_fixes) - 1 if len(join_fixes) != 1 else 1000 #тк ничего не сделал если слово 1
    results[1] = PrioritizedItem((count_bigrams(join_fixes, language_model) / tmp, \
                                  count_bigrams(join_fixes, language_model) / tmp, \
                                  count_unigrams(join_fixes, clear_language_model) / len(join_fixes), \
                                  count_unigrams(join_fixes, language_model) / len(join_fixes)),\
                                 (join_fixes))
    cond = int(len(fixes) == count_unigrams(fixes, language_model))
    tmp = len(fixes) - 1 if len(fixes) != 1 else 1
    results[0] = PrioritizedItem((count_bigrams(fixes, language_model) / tmp, \
                                  count_bigrams(fixes, language_model) / tmp, \
                                  count_unigrams(fixes, clear_language_model) / len(fixes), \
                                  count_unigrams(fixes, language_model) / len(fixes)), (fixes))
    results = sorted(results, reverse=True)
    return ' '.join(results[0].item)

def main():
    with open('language_model', 'rb') as f:
        language_model = pickle.load(f)
    with open('error_model', 'rb') as f:
        error_model = pickle.load(f)
    with open('clear_language_model', 'rb') as f:
        clear_language_model = pickle.load(f)
    with open('model', 'rb') as f:
        model = pickle.load(f)
    trie = Trie()
    split = SplitFixes(language_model)
    join = JoinFixes(language_model, clear_language_model)
    keyboard = Keyboard(language_model)
    trie.CreateTrie(model, language_model, error_model)
    for query in sys.stdin:
        print(get_fix(query, trie, keyboard, split, join, language_model, clear_language_model))

if __name__ == "__main__":
    main()

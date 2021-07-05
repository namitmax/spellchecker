import operator
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import heapq

def get_query_proba(query, language_model):
    f = 0
    prob = 1
    for i in range(len(query) - 1):
        tmp = language_model.GetWeight(query[i], query[i + 1])
        prob *= tmp
        if (tmp > 1e-10):
            f = 1
    tmp = language_model.GetWeight(query[-1], '')
    if (tmp > 1e-10):
        f = 1
    return prob * tmp * f

class CandidateElement:
    def __init__(self, word, weight, \
                 fixed_w, fix_lev_w, fix_w, freq_w, lev_distance, lev_prob):
        self.word = word
        self.weight = weight
        self.fixed_w = fixed_w
        self.fix_lev_w = fix_lev_w
        self.fix_w = fix_w
        self.freq_w = freq_w 
        self.lev_distance = lev_distance
        self.lev_prob = lev_prob
    
    def __str__(self):
        return 'word = ' + str(self.word) + ' weight ' + str(self.weight) + ' fixed_w ' \
                + str(self.fixed_w) + ' fix_lev_w ' + str(self.fix_lev_w) + ' fix_w ' +  str(self.fix_w) \
                + ' freq_w ' + str(self.freq_w) + ' lev_dist ' +  str(self.lev_distance) \
                + ' lev_prob ' + str(self.lev_prob) + '\n'

def get_fixes(trie, word, language_model, bad_els_length, a=1.0, b=0., c=0., d=1., e=0., \
              w1 = 0, word_1 = '', w2 = 0, word_2 = '', clear_language_model = None):
    res = []
    bad_list = []
    alpha = 1.5
    #50, 10, 1e+3, 0, 1e-2, 1e-3
    for i, w, word in trie.get_fixed_query(word, 30, 10, 1e+3, 0, 1e-2, 1e-4, \
                                           alpha, 50, bad_list, 3e+3):
        freq = language_model.GetWeight(i, '')
        prob = (1 / alpha) ** (w[3])
        res.append(CandidateElement(i, a * w[0] + b * w[1] + c * w[2] + d * freq + e * prob \
                                       + w1 * clear_language_model.GetWeight(i, word_1) * freq \
                                       + w2 * clear_language_model.GetWeight(word_2, i) \
                                       * clear_language_model.GetWeight(word_2, '') , \
                                    w[0], w[1], w[2], freq, w[3], prob))
        bad_list.append(word)
    return res

def generate_candidates(trie, inp, language_model, clear_language_model, \
                        nice_els, bad_els):
    candidates = []
    num_cand = 30
    query_length = len(inp)
    for i in range(query_length):
        candidates.append([])
    for i in bad_els:
        if (((i + 1) in bad_els) or (i + 1 >= query_length)):
            w1 = 0
            word_1 = ''
        else:
            w1 = 1e+4
            word_1 = inp[i + 1]
        if (((i - 1) in  bad_els) or (i - 1 <= 0)):
            w2 = 0
            word_2 = ''
        else:
            w2 = 1e+4
            word_2 = inp[i - 1]
        temp = get_fixes(trie, inp[i], language_model, len(bad_els),\
                         0, 0, 0, 1e+4, 1., w1, word_1, w2, word_2, clear_language_model)
        temp.sort(key=operator.attrgetter('weight'), reverse=True)
        num = min(num_cand, len(temp))
        candidates[i] = temp[:num]
    for i, cand in enumerate(candidates):
        if cand == []:
            candidates[i] = [CandidateElement(inp[i], 0, 0, 0, 0, 0, 0, 0)]
    return candidates, bad_els, nice_els
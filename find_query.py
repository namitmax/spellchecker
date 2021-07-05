import operator
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import heapq

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)

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

def get_best(temp_results, language_model, clear_language_model):
    max_val = -1
    best = 0
    for num, query in enumerate(temp_results):
        temp_lev_dist = 0
        temp_bigram_clear_w = 1
        temp_uniram_clear_w = 1
        temp_bigram_w = 1
        temp_uniram_w = 1
        l = len(query)
        for i, element in enumerate(query):
            word = element.word
            next_word = query[i + 1].word if i < l - 1 else ''
            #### clear model ###
            prob_bi_clear = clear_language_model.GetWeight(word, next_word)
            prob_uni_clear = clear_language_model.GetWeight(word, '')
            temp_bigram_clear_w *= prob_bi_clear
            temp_uniram_clear_w *= prob_uni_clear
            ### model ###
            prob_bi = language_model.GetWeight(word, next_word)
            prob_uni = language_model.GetWeight(word, '')
            temp_bigram_w *= prob_bi
            temp_uniram_w *= prob_uni
            ##############
            temp_lev_dist += element.lev_distance
        #print(temp_bigram_clear_w, temp_bigram_w)
        result_val = 10 * temp_bigram_clear_w + temp_bigram_w 
        #print([word.word for word in temp_results[num]], result_val, temp_lev_dist)
        if (result_val > max_val):
            max_val = result_val
            best = num
    return [word.word for word in temp_results[best]]

def get_nice_query(founder, nice_els, num=5, step=0):
    l = len(founder.query_result)
    cands = founder.query_result
    best  = []
    for i, cand in enumerate(cands):
        tmp = cand[step].item if len(cand) > step else cand[-1].item
        if i > nice_els[0] and len(tmp) > 1:
            best.append(tmp[1])
            continue
        best.append(tmp[0])
    return best

def get_bad_query(founder, num=5, step=0):
    l = len(founder.query_result)
    cands = founder.query_result
    best  = []
    for i, cand in enumerate(cands):
        if i == 0:
            tmp = cands[i+1][step].item[0] \
                    if len(cands[i+1]) > step else\
                    cands[i+1][-1].item[0]
            best.append(tmp)
            continue
        tmp = cand[step].item[1] if len(cand) > step else cand[-1].item[1]
        best.append(tmp)
    return best

def get_best_queries(founder, nice_els, num=5, is_bad=True):
    l = len(founder.query_result)
    step = 0
    result = []
    for step in range(num):
        if l == 1:
            tmp = founder.query_result[0][step] if step < len(founder.query_result[0]) \
                  else founder.query_result[0][-1]
            result.append([tmp.item[1]])
            continue
        if is_bad:
            result.append(get_bad_query(founder, num, step))
            continue
        result.append(get_nice_query(founder, nice_els, num, step))
    return result

class FindBestQuery:
    def __init__(self, inp, candidates, language_model, clear_language_model, nice_els, bad_els, num = 10):
        self.query_length = len(inp)
        self.input = inp
        self.candidates = candidates
        self.query_result = []
        for i in range(self.query_length):
            self.query_result.append([])
        self.bad_els = bad_els
        self.nice_els = nice_els
        self.num = num
        self.all = []
        self.language_model = language_model
        self.clear_language_model = clear_language_model

    def WeightFunc(self, first, second):
        bi_prob = self.language_model.GetWeight(first.word, second.word)
        uni_prob = self.language_model.GetWeight(first.word, '')
        bi_clear_prob = self.clear_language_model.GetWeight(first.word, second.word)
        uni_clear_prob = self.clear_language_model.GetWeight(first.word, '')
        return bi_prob * uni_prob + 10 * bi_clear_prob * uni_clear_prob
    
    def BackPass(self):
        if len(self.nice_els) != 0 and self.nice_els[0] != 0:
            next_elements = [PrioritizedItem(-1, ([self.candidates[self.nice_els[0]][0]]))]
            self.query_result[self.nice_els[0]] = next_elements
            l = len(self.candidates[:self.nice_els[0]])
            delta = self.query_length - l + 1
            for i, candidate in enumerate(reversed(self.candidates[:self.nice_els[0]])):
                dists = []
                for next_el_pr in next_elements:
                    next_el = next_el_pr.item[0]
                    dists.extend(list(map( lambda first: \
                        PrioritizedItem(self.WeightFunc(first, next_el), (first, next_el)),\
                        self.candidates[self.query_length - delta - i])))
                next_elements = heapq.nlargest(self.num, dists)
                self.query_result[self.query_length - delta - i] = next_elements

    def ForwardPass(self):
        for i, nice_el in enumerate(self.nice_els):
            prev_elements = [PrioritizedItem(-1, ([self.candidates[nice_el][0]]))]
            if i < len(self.nice_els) - 1:
                cands = self.candidates[self.nice_els[i] + 1 : self.nice_els[i + 1]]
            else:
                cands = self.candidates[self.nice_els[i] : -1]
            self.query_result[nice_el] = [PrioritizedItem(-1, ([self.candidates[nice_el][0]]))]
            for j, cand in enumerate(cands):
                dists = []
                for prev_el_pr in prev_elements:
                    prev_el = prev_el_pr.item[0]
                    dists.extend(list(map(lambda second:\
                        PrioritizedItem(self.WeightFunc(prev_el, second), (prev_el, second, True)),\
                        self.candidates[nice_el + j + 1])))
                prev_elements = heapq.nlargest(self.num, dists)
                self.query_result[nice_el + j + 1] = prev_elements
    
    def BadCasePass(self):
        prev_elements = [PrioritizedItem(self.language_model.GetWeight('<', el.word) + \
                            10 * self.clear_language_model.GetWeight('<', el.word), \
                            (el, el)) \
                         for el in self.candidates[0]]
        self.query_result[0] = prev_elements
        for i, candidate in enumerate(self.candidates):
            if i == 0:
                continue
            dists = []
            for prev_el_pr in prev_elements:
                prev_el = prev_el_pr.item[1]
                prob = self.language_model.GetWeight(prev_el.word, '')
                prob_clear = self.clear_language_model.GetWeight(prev_el.word, '')
                dists.extend(list(map( lambda second: \
                    PrioritizedItem(self.WeightFunc(prev_el, second), (prev_el, second)),\
                    candidate)))
            prev_elements = heapq.nlargest(self.num, dists)
            self.query_result[i] = prev_elements                
    
    def get_ord_fixes(self):
        if len(self.nice_els) != 0:
            self.BackPass()
            self.ForwardPass()
            return
        self.BadCasePass()
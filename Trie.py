import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)

class TrieIterator:

    def goBack(self, node, s):
        sRes = s
        while node.previous != {}:
            ch = next(iter(node.previous))
            sRes += ch
            node = node.previous[ch]
        return sRes[::-1]

    def __init__(self, start, word, n_find, language_model, error_model, _all, \
                 a = 1.0, b = 1.0, c = 1.0, stopIt = 1e-5, stopFix=1e-20, alpha=2.0, num_letters=10, bad_list=[], stop=1e+10):
        self.t = []
        self.max = n_find
        self.all = _all
        self.found = 0
        self.word = word
        self.word_counter = len(word)
        self.word_len = self.word_counter
        self.language_model = language_model
        self.num_letters = num_letters
        self.error_model = error_model
        self.result = (bad_list)
        self.a = a
        self.b = b
        self.c = c
        self.stopIt = stopIt
        self.stopALL = stop
        self.numIter = 0
        self.alpha = 1.0 / alpha
        self.stopFix = stopFix
        tmp = self.word + '_'
        #self.error_model.GetWeight(tmp[i:i+2], '_' + tmp[i+1:i+2])
        #self.delete_weights = [self.error_model.GetWeight(tmp[i:i+2], '_' + tmp[i+1:i+2])\
        #                      for i, _ in enumerate(self.word)]
        temp = (self.word_len / 2 + 1)# if self.word_len < 6 else 3
        self.stopFixLev = temp
        self.letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' +\
                        'abcdefghijklmnopqrstuvwxyz' + '0123456789'
        init_lev_distance = 0#sum(self.delete_weights)
        heapq.heappush(self.t, PrioritizedItem(1., (start, ('~', ''), len(word), 0, 1., 1, init_lev_distance)))

    def __iter__(self):
        return self
    
    def GetWeight(self, vertex, temp_lev_distance, counter):
        prefix_prob = self.alpha ** (temp_lev_distance)
        prefix_freq = np.log2(vertex.count)
        tmp = counter / (self.word_counter - counter + 1) 
        tmp = tmp if tmp > 0 else 1
        #print('freq w =', self.a * prefix_freq * prefix_freq, \
        #      ' prob w =', self.b *np.log2(1 + prefix_prob), \
        #      'res =', self.a * prefix_freq + self.b *np.log2(1 + prefix_prob))
        return (self.a * prefix_freq + self.b *np.log2(1 + prefix_prob)), prefix_prob
    
    def Replace(self, local_t, prefix, i, temp, counter, temp_dist, priority_prev, \
                temp_lev_distance):
        pref = prefix[0] + i
        #print('replace ', prefix, pref)
        temp_dist = temp_dist + 1 if pref != prefix else temp_dist
        #temp_lev_distance -= self.delete_weights[self.word_len - counter]
        #############
        weight_replace = self.error_model.GetWeight(prefix, pref) # 'aе' -> 'ое'
        temp_lev_distance += weight_replace
        #############
        #print('replace ', prefix, pref, temp_lev_distance)
        weight, prefix_prob = self.GetWeight(temp.children[i], temp_lev_distance, counter) 
        #############
        #print('replace', 1 - weight_replace)
        local_t.append(PrioritizedItem(-(weight), \
                        (temp.children, (i, ''), counter - 1, temp_dist, \
                         (1 - weight_replace) * priority_prev, prefix_prob, temp_lev_distance)))

    def Insert(self, local_t, prefix, i, temp, let, counter, temp_dist, priority_prev, \
              temp_lev_distance):
        temp_dist += 1
        #print('insert ', prefix[0] + '_', prefix[0] + i)
        #############
        weight_insert = self.error_model.GetWeight(prefix[0] + '_', prefix[0] + i)
        temp_lev_distance += weight_insert
        #############
        #print('insert ', '_' + prefix[0], i + prefix[0], temp_lev_distance)
        weight, prefix_prob = self.GetWeight(temp.children[i], temp_lev_distance, counter)
        ##############
        #print('insert', 1 - weight_insert)
        local_t.append(PrioritizedItem(-(weight), \
                        (temp.children, (i, ''), counter, temp_dist, \
                         (1 - weight_insert) * priority_prev, prefix_prob, temp_lev_distance)))
            
    def Delete(self, local_t, prefix, temp, last, counter, temp_dist, priority_prev, \
               temp_lev_distance):
        temp_dist += 1
        #print('delete ', prefix, prefix[0] + '_')
        #############
        weight_delete = self.error_model.GetWeight(prefix, prefix[0] + '_')
        temp_lev_distance += weight_delete
        #############
        #print('delete ', prefix, '_' + prefix[1], temp_lev_distance)
        weight, prefix_prob = self.GetWeight(temp, temp_lev_distance, counter)
        #print('delete', 1 - weight_delete)
        #############
        local_t.append(PrioritizedItem(-(weight), \
                           (temp, ('', last), counter - 1, temp_dist, \
                            (1 - weight_delete) * priority_prev, prefix_prob, temp_lev_distance)))

    def __next__(self):
        #MAX = copy.deepcopy(self.stopALL)
        if len(self.t) != 0 and self.found < self.max:
            #print(len(self.t))
            while (len(self.t) != 0 and self.word_counter > -2):
                temp_element =  heapq.heappop(self.t)
                cond = abs(temp_element.priority) # priority of temporary element
                temp_dist = temp_element.item[3]
                priority_prev = temp_element.item[4]
                pre_weight = temp_element.item[5]
                temp_lev_distance = temp_element.item[6]
                #print('========')
                #print(temp_element.item[-1], temp_lev_distance, len(temp_element.item))
                #print('!', cond, pre_weight, priority_prev, len(self.t), temp_element.item[2], temp_lev_distance)
                #cond, pre_weight, temp_element.item[2],
                if (self.numIter > self.stopALL): #cond < self.stopIt or 
                    #print('!!')
                    raise StopIteration()
                if (priority_prev < self.stopFix or cond < self.stopIt or temp_dist > self.stopFixLev):
                    #print('!!!!! ', priority_prev, self.stopFix, \
                    #                 cond, self.stopIt,\
                    #                  temp_dist, self.stopFixLev)
                    continue
                ################
                local_t = []
                temp_dict = temp_element.item[0]
                let = temp_element.item[1][0]
                word_counter = temp_element.item[2]
                self.word_counter = word_counter
                num = self.word_len - self.word_counter
                #print('====')
                #print('! ', num, temp_element.priority)
                temp = temp_dict[let] if (let != '' and let != '~') else temp_dict
                let = let if let != '' else temp_element.item[1][1]
                if (num != 0):#self.word_len - 1):
                    prefix = self.word[num - 1: num + 1]
                else:
                    prefix = '<' + self.word[0]
                ################
                #print('!', prefix)
                if (self.word_counter > 0):
                    for i in self.letters:
                        if i in (temp.children):
                            self.Replace(local_t, prefix, i, temp, \
                                         word_counter, temp_dist, priority_prev, \
                                         temp_lev_distance)
                            self.Insert(local_t, prefix, i, temp, \
                                        let, word_counter, temp_dist, priority_prev, \
                                        temp_lev_distance)
                    self.Delete(local_t, prefix, temp, \
                                let, word_counter, temp_dist, priority_prev, \
                                temp_lev_distance)
                elif (self.word_counter == 0):
                    for i in self.letters:
                        if i in (temp.children):
                            self.Insert(local_t, prefix, i, temp, let, self.word_counter - 1, \
                                        temp_dist, priority_prev, temp_lev_distance)
                _min = min(len(local_t), self.num_letters)
                tmp_t = heapq.nsmallest(_min, local_t)
                #print('==========')
                #print(prefix)
                for el in tmp_t:
                    #print(el.priority)
                    heapq.heappush(self.t, el)
                if (temp not in self.result and (word_counter == -1 or word_counter == 0) \
                    and temp.isEndOfWord):
                    #print('========')
                    #print(temp.isEndOfWord)
                    result_weight = pre_weight
                    self.result.append(temp)
                    a = self.goBack(temp, let)
                    self.found += 1
                    return a, (result_weight, priority_prev, cond, temp_lev_distance), temp
                self.numIter += 1
                #print(word_counter)
            raise StopIteration()
        else:
            raise StopIteration()

class TrieNode:
    def __init__(self):
        self.children = {}
        self.previous = {}
        self.count = 0
        self.isEndOfWord = False
    
    def __str__(self):
        return 'count = ' + str(self.count)
        
class TrieNode:
    def __init__(self):
        self.children = {}
        self.previous = {}
        self.count = 0
        self.isEndOfWord = False
    
    def __str__(self):
        return 'count = ' + str(self.count)
        
class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.root.count = 1
        self.all_prefixes = 0
        self.deep = 0
    
    def __len__(self):
        return self.deep
    
    def get_fixed_query(self, word, n_find, a = 1.0, b = 1.0, c = 1.0, stopIt = 1e-5, stopFix=1e-20, alpha=2.0, \
                        num_letters = 10, bad_list=[], stop=1e+20):
        return TrieIterator(self.root, word, n_find, self.language_model, self.error_model, self.all_prefixes, \
                            a, b, c, stopIt, stopFix, alpha, num_letters, bad_list, stop)

    def add(self, word: str) -> None:
        if word == '':
            return
        self.deep += 1
        node = self.root
        t = ''
        for i in range(len(word)):
            char = word[i]
            if char not in node.children:
                node.children[char] = TrieNode()
                node.children[char].previous[t] = node
                node.children[char].count = 1
                self.all_prefixes += 1
            else:
                node.children[char].count += 1
                self.all_prefixes += 1
            t = char
            node = node.children[char]
        node.isEndOfWord = True
    
    def CreateTrie(self, queries, language_model, error_model) -> None:
        self.language_model = language_model
        self.error_model = error_model
        for query in queries:
            for word in query:
                self.add(word)
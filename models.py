import re
import Levenshtein

class LanguageModel():
    def __init__(self):
        self.wordsCounter = dict()
        self.len = 0

    def appendWord(self, word, next_word):
        word = word.lower()
        next_word = next_word.lower()
        if word in self.wordsCounter:
            self.wordsCounter[word][''] += 1
            if next_word in self.wordsCounter[word]:
                self.wordsCounter[word][next_word] += 1
            else:
                self.wordsCounter[word][next_word] = 1
        else:
            self.wordsCounter[word] = dict()
            self.wordsCounter[word][''] = 1
            self.wordsCounter[word][next_word] = 1
        self.len += 1

    def CountsToWeights(self):
        for word in self.wordsCounter:
            for second_word in self.wordsCounter[word]:
                if second_word == '':
                    continue
                self.wordsCounter[word][second_word] /= self.wordsCounter[word]['']
            self.wordsCounter[word][''] /= self.len

    def GetWeight(self, word1, word2):
        if word1 in self.wordsCounter:
            if word2 in self.wordsCounter[word1]:
                return self.wordsCounter[word1][word2]
        return 1e-10

class ErrorModel():
    def __init__(self):
        self.errorsCounter = dict()
        self.type_of_fix = dict({'insert'  : self.fix_insert,
                                 'delete'  : self.fix_delete,
                                 'replace' : self.fix_replace,
                                 'equal' :   self.empty})
        self.all = 0

    def empty(*args):
        pass
    
    def append_prefix(self, prefix):
        if prefix in self.errorsCounter:
            self.errorsCounter[prefix]['-'] += 1
        else:
            self.errorsCounter[prefix] = dict()
            self.errorsCounter[prefix]['-'] = 1
        self.all +=1
    
    def append_prefix_fix(self, pre_or, pre_fix):
        self.append_prefix(pre_or)
        self.append_prefix(pre_fix)
        if pre_or in self.errorsCounter:
            if pre_fix in self.errorsCounter[pre_or]:
                self.errorsCounter[pre_or][pre_fix] += 1
            else:
                self.errorsCounter[pre_or][pre_fix] = 1
        else:
            self.append_prefix(pre_or)
            self.errorsCounter[pre_or][pre_fix] = 1

    def fix_insert(self, _, fix):
        for i in range(1, len(fix) - 1):
            original = fix[:i] + '_' + fix[i+1:]
            pre_fix_1 = fix[i-1:i+1]
            pre_or_1 = original[i-1:i+1]
            pre_fix_2 = fix[i:i+2]
            pre_or_2 = original[i:i+2]
            #print(pre_or_2, pre_fix_2)
            self.append_prefix_fix(pre_or_1, pre_fix_1)
            self.append_prefix_fix(pre_or_2, pre_fix_2)
            #self.all +=1

    def fix_delete(self, original, _):
        for i in range(1, len(original) - 1):
            fix = original[:i] + '_' + original[i+1:]
            pre_or_1 = original[i-1:i+1]
            pre_fix_1 = fix[i-1:i+1]
            pre_or_2 = original[i:i+2]
            pre_fix_2 = fix[i:i+2]
            #print(pre_or_2, pre_fix_2)
            self.append_prefix_fix(pre_or_1, pre_fix_1)
            self.append_prefix_fix(pre_or_2, pre_fix_2)
            #self.all +=1

    def fix_replace(self, original, fix):
        for i in range(1, len(original) - 1):
            pre_or = original[i-1:i+1]
            pre_fix = fix[i-1:i+1]
            pre_fix_1 = fix[i-1:i+1]
            pre_or_1 = original[i-1:i+1]
            pre_fix_2 = fix[i:i+2]
            pre_or_2 = original[i:i+2]
            #print(pre_or_2, pre_fix_2)
            self.append_prefix_fix(pre_or_1, pre_fix_1)
            self.append_prefix_fix(pre_or_2, pre_fix_2)
            #self.all +=1

    def appendFixesForQuery(self, query, keyboard):
        original_words = query[0]
        fix_words = query[1]
        if (len(original_words) != len(fix_words)):
            return
        for i in range(min(len(original_words), len(fix_words))):
            original = original_words[i].lower()
            fix = fix_words[i].lower()
            en_var_original = keyboard.to_eng(original)
            ru_var_original = keyboard.to_rus(original)
            if (fix == en_var_original or fix == ru_var_original):
                continue
            original = '<' + original + '>'
            fix = '<' + fix + '>'
            operations = Levenshtein.opcodes(original, fix)
            #print('!! ======= !!')
            #print(original, fix)
            for operation in operations:
                operation_type = operation[0]
                begin_origin, begin_fix = operation[1], operation[3]
                end_origin, end_fix = operation[2], operation[4]
                fix_count_func = self.type_of_fix[operation_type]
                fix_count_func(original[begin_origin - 1: end_origin + 1], \
                               fix[begin_fix - 1: end_fix + 1])

    def CountsToWeights(self):
        for el in self.errorsCounter:
            for second_el in self.errorsCounter[el]:
                if second_el == '-':
                    continue
                self.errorsCounter[el][second_el] = (1 - self.errorsCounter[el][second_el] \
                                                     / self.errorsCounter[el]['-'])
            self.errorsCounter[el]['-'] = (1 - self.errorsCounter[el]['-'] / self.all)

    def GetWeight(self, prefix1, prefix2):
        if prefix1 == prefix2:
            if prefix1 in self.errorsCounter:
                el_min = min(self.errorsCounter[prefix1], \
                               key=lambda x: self.errorsCounter[prefix1][x])
                return 0
            else:
                return 0
        if prefix1 in self.errorsCounter:
            if prefix2 in self.errorsCounter[prefix1]:
                return self.errorsCounter[prefix1][prefix2]
        return 1

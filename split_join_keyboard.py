import numpy as np

class JoinFixes:

    def __init__(self, language_model, clear_language_model):
        self.language_model = language_model
        self.clear_language_model = clear_language_model

    def get_bigram_weight(self, words):
        l = len(words)
        prob = 1.0
        f = 0
        for i in range(1, l):
            tmp = self.language_model.GetWeight(words[i-1], words[i])
            if (tmp > 1e-10):
                f = 1
            prob *= tmp
        return prob, f
    
    def get_unigram_weight(self, words):
        prob = 1.0
        f = 0
        for word in words:
            tmp = self.language_model.GetWeight(word, '')
            if (tmp > 1e-10):
                f = 1
            prob *= tmp
        return prob, f
        
    def get_join_fix_word_forward(self, word, third_word):
        new_word_1 = ''
        new_word_2 = ''
        best_query = [word, third_word]
        best_prob, zero = self.get_bigram_weight(best_query)
        unigram = False
        fix = False
        if (zero == 0):
            best_prob, zero = self.get_unigram_weight(best_query[0:-1])
            best_prob *= zero
            unigram = True
        word_l = len(word)
        for i in range(1, word_l):
            cand_word_1 = word[:i]
            cand_word_2 = word[i:]
            words = [cand_word_1, cand_word_2, third_word]
            if (self.language_model.GetWeight(cand_word_1, '') < 1.1e-10 or
                self.language_model.GetWeight(cand_word_2, '') < 1.1e-10):
                continue
            val_0, zero_0 = self.get_bigram_weight(words)
            val_0 *= zero_0
            if (zero_0 == 0 and unigram):
                val_0, zero_0 = self.get_unigram_weight(words[0:-1])
            else:
                if (unigram):
                    best_prob = 0
                unigram = False
            #print(words, val_0, best_prob, unigram)
            if val_0 > best_prob:
                best_prob = val_0
                new_word_1 = cand_word_1
                new_word_2 = cand_word_2
                best_query = words
                fix = True
        return best_query, fix

    def get_join_fix_forward(self, word, third_word):
        temp_result = []
        best_query, fixed \
            = self.get_join_fix_word_forward(word, third_word)
        #print(best_query, fixed)
        for word in best_query:
            temp_result.append(word)
        return temp_result, fixed

    def fix_join(self, query):
        l = len(query)
        new_query = []
        i = 0
        while i < l:
            #print(i, l)
            if self.clear_language_model.GetWeight(query[i], '') > 1e-10:
                i += 1
                continue
            if (i + 1 < l and l != 1):
                new_query, fixed = self.get_join_fix_forward(query[i], query[i + 1])
            else:
                #print('!')
                new_query, fixed = self.get_join_fix_forward(query[i], '>')
            #print(new_query, query[:i], query[i+1:])
            if fixed:
                query = query[:i] + new_query[:-1] + query[i + 1:]
                l += 1
                i += 2
                #print(query, i, l)
                continue
            fixed = False
            i += 1
        return query

class SplitFixes:

    def __init__(self, language_model):
        self.language_model = language_model

    def check_zero(self, val, deg=1):
        zero = (1e-10)**deg
        return (abs(zero - val) < (zero/10.0))
    
    def get_bigram_weight(self, words):
        l = len(words)
        prob = 1.0
        f = 0
        for i in range(1, l):
            tmp = self.language_model.GetWeight(words[i-1], words[i])
            if (tmp > 1e-10):
                f = 1
            prob *= tmp
        return prob, f
    
    def get_unigram_weight(self, words):
        prob = 1.0
        f = 0
        for word in words:
            tmp = self.language_model.GetWeight(word, '')
            #print(word, tmp)
            if (tmp > 1e-10):
                f = 1
            prob *= tmp
        return prob, f
    
    def fix_split(self, query):
        l = len(query)
        new_query = []
        skip = False
        i = 0
        vals = [0,0]
        if l == 1:
            return query
        if l == 2:
            words_0 = ['<', query[0], query[1], '>']
            words_1 = ['<', query[0] + query[1], '>']
            vals[0], zero_0 = self.get_bigram_weight(words_0)
            vals[1], zero_1 = self.get_bigram_weight(words_1)
            if (zero_1 == 0 and zero_0 != 0 \
                or self.language_model.GetWeight(query[0] + query[1], '') < 1.1e-10):
                return query
            if (zero_0 == 0 and zero_1 == 0):
                vals[0], zero_0 = self.get_unigram_weight(words_0)
                vals[0] *= zero_0
                vals[1], zero_1 = self.get_unigram_weight(words_1)
                vals[1] *= zero_1
            #print(vals)
            min_val = np.argmax(vals)
            if min_val == 0 \
                or (zero_0 == 0 and zero_1 == 0):
                return query
            else:
                return [query[0] + query[1]]
        vals = [0,0,0]
        while i < l - 2:
            #print(query[i], query[i + 1], query[i + 2])
            #print('!!!', new_query)
            words_0 = [query[i], query[i + 1], query[i + 2]]
            words_1 = [query[i] + query[i + 1], query[i + 2]]
            #if :
            #    i += 1
            #   new_query.append(query[i])
            #    continue
            words_2 = [query[i], query[i + 1] + query[i + 2]]
            vals[0], zero_0 = self.get_bigram_weight(words_0)
            vals[1], zero_1 = self.get_bigram_weight(words_1)
            vals[2], zero_2 = self.get_bigram_weight(words_2)
            if (zero_0 == 0 and zero_1 == 0 and zero_2 == 0):
                #print('!')
                vals[0], zero_0 = self.get_unigram_weight(words_0[0:-1])
                vals[0] *= zero_0
                vals[1], zero_1 = self.get_unigram_weight(words_1[0:-1])
                vals[1] *= zero_1
                vals[2] = 0
            #print(vals, )
            min_val = np.argmax(vals)
            if (zero_1 == 0 and zero_2 == 0) or \
               (self.language_model.GetWeight(query[i] + query[i + 1], '') < 1.1e-10):
                min_val = 0
            if min_val == 1:
                new_query.append(query[i] + query[i + 1])
                query = new_query + query[i + 2:]
                skip = True
                l -= 1
                if i >= l - 2:
                    new_query.append(query[i + 1])
                continue
            if not skip:
                new_query.append(query[i])
            if i + 1 >= l - 2:
                if min_val == 0:
                    new_query.append(query[i + 1])
                    new_query.append(query[i + 2])
                else:
                    new_query.append(query[i + 1] + query[i + 2])
            skip = False
            i += 1
        return new_query
    
class Keyboard:
    def __init__(self, language_model):
        self.en = "qwertyuiop[]asdfghjkl;'zxcvbnm,.`"
        self.ru = "йцукенгшщзхъфывапролджэячсмитьбюё"
        self.language_model = language_model

    def to_eng(self, word):
        new_word = word
        for i, char in enumerate(word):
            if char in self.ru:
                new_word = new_word[:i] + self.en[self.ru.find(char)] + new_word[i+1:]
        return new_word

    def to_rus(self, word):
        new_word = word
        for i, char in enumerate(word):
            if char in self.en:
                new_word = new_word[:i] + self.ru[self.en.find(char)] + new_word[i+1:]
        return new_word

    def change_keyboard(self, word):
        en_word = self.to_eng(word)
        rus_word = self.to_rus(word)
        if self.language_model.GetWeight(rus_word, '') >= self.language_model.GetWeight(en_word, ''):
            return rus_word
        return en_word

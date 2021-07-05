import re
from collections import defaultdict
from models import LanguageModel
from models import ErrorModel
from string import punctuation
import copy
from tqdm import tqdm
import pickle
from split_join_keyboard import Keyboard

def check_query(query_0, query_1, keyboard):
    if len(query_0) != len(query_1):
        return False
    for i in range(len(query_0)):
        if (query_0[i] != query_1[i]):
            en_var = keyboard.to_eng(query_0[i])
            ru_var = keyboard.to_rus(query_0[i])
            if (query_1[i] != en_var and query_1[i] != ru_var):
                return True
    return False

def process_file(filename):
    model = []
    fixes = []
    keyboard = Keyboard(None)
    with open(filename, 'r', encoding='utf-8') as f:
        file = f.readlines()
        j = 0
        for line in tqdm(file):
            query = line.strip('\n')
            query = query.split('\t')
            for i, _ in enumerate(query):
                query[i] = re.sub('[' + punctuation + ']', '', query[i]).split()
            if len(query) != 2:
                model.append(query[0])
                #for word in query[0]:
                #    tmp = '_' + word + '_'
                #    for i in range(len(tmp)-1):
                #        error_model.append_prefix(tmp[i:i+2])
            else:
                if not check_query(query[0], query[1], keyboard):
                    continue
                fixes.append(copy.copy(query))
                model.append(query[1])
                j += 1
    return model, fixes

def CreateLanguageModel(words_model):
    model = LanguageModel()
    for query in tqdm(words_model):
        l = len(query)
        for i, word in enumerate(query):
            if i == 0:
                model.appendWord('<', word)
            if (i + 1 < l):
                model.appendWord(word, query[i + 1])
            else:
                model.appendWord(word, '>')
    model.CountsToWeights()
    return model

def CreateClearLanguageModel(words_model):
    model = LanguageModel()
    for q in tqdm(words_model):
        query = q[1]
        l = len(query)
        for i, word in enumerate(query):
            if i == 0:
                model.appendWord('<', word)
            if (i + 1 < l):
                model.appendWord(word, query[i + 1])
            else:
                model.appendWord(word, '>')
    model.CountsToWeights()
    return model

def CreateErrorModel(words_fixes):
    model = ErrorModel()
    keyboard = Keyboard(None)
    for query in tqdm(words_fixes):
        model.appendFixesForQuery(query, keyboard) 
    model.CountsToWeights()
    return model

def main():
    model = []
    fixes = []
    #error_model = ErrorModel()
    model, fixes = process_file("queries_all.txt")
    language_model = CreateLanguageModel(model)
    clear_language_model = CreateClearLanguageModel(fixes)
    error_model = CreateErrorModel(fixes)
    with open("model", "wb") as f:
        pickle.dump(model, f)
    with open("language_model", "wb") as f:
        pickle.dump(language_model, f)
    with open("clear_language_model", "wb") as f:
        pickle.dump(clear_language_model, f)
    with open("error_model", "wb") as f:
        pickle.dump(error_model, f)

if __name__ == "__main__":
    main()

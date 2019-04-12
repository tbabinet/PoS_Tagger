import os
from collections import defaultdict
import json
import math
# import kenlm


class NoisinessMeasurer:
    def __init__(self, train_set, test_set):
        self.train_set  = train_set

        self.test_set = test_set
        self.train_vocab = self.make_vocab(train_set)
        self.test_vocab = self.make_vocab(test_set)
        self.train_alphabet = self.make_alphabet(train_set)
        self.test_alphabet = self.make_alphabet(test_set)
        self.oovw = self.test_vocab - self.train_vocab #Ensemble des mots qu ne sont pas dans le vocabulaires de train
        self.KLD = self.KL_divergence()

    def make_vocab(self, set_):
        rep = set()
        for sent, _ in set_:
            for w in sent:
                rep.add(w)
        return rep
    
    def make_alphabet(self,set_):
        rep = dict()
        for sent, _ in set_:
            for w in sent:
                for c in w:
                    if c in rep:
                        rep[c]+=1
                    else:
                        rep[c] = 1
        return rep
    
    

    """
    Renvoie un dictionnaire contenant le compte de tous les trigrames de caractères
    de la liste de strings (liste de phrases) passée en paramètre
    """
    
    def char_trigrams_dict(self,string_list):
        rep = defaultdict(int)
        for substring in string_list: 
            for i in range(2,len(substring)-1):
                trigram = substring [i-2:i+1]
                if(trigram in rep):
                    rep[trigram]+=1
                else:
                    rep[trigram]=1
        return rep
    
    
    """
    transforme un ensemble (train, dev, test) en une liste de string, ou chaque string est la concaténation de l'ensemble de mots constituant une phrase
    Utilisé pour la construction du dictionnaires de 3-gram de caractères
    """
    def set_to_string_list(self,set_):
        rep = list()
        for sent, _ in set_:
            string =""
            for w in sent:
                string+=w
            rep.append(string)
        return rep
    
    """
    Calcul de la KL-divergence, prend en entrée l'ensemble de train, de test, ainsi que les alphabets respectifs.
    """
    def KL_divergence(self):
        print(len(self.train_set), len(self.test_set))
        str_train_set = self.set_to_string_list(self.train_set)
        str_test_set = self.set_to_string_list(self.test_set)

        train_3grams = self.char_trigrams_dict(str_train_set)
        test_3grams = self.char_trigrams_dict(str_test_set)

        merged_dict = {**train_3grams, **test_3grams} #V

        total3grams = sum(train_3grams[k] for k in train_3grams)#N
        total3grams+=sum(test_3grams[k] for k in test_3grams)
        
        total_train_chars = sum(self.train_alphabet[k] for k in self.train_alphabet)# #train
        total_test_chars = sum(self.test_alphabet[k] for k in self.test_alphabet) # #test
        
        res = 0
        """
        on calcul ptest et ptrain en utilisant le nombre d'apparitions du trigram au carré plutôt que de refaire un calcul pour chaque apparition, ce qui 
        reviendrait au même au final
        
        """
        for trigram in merged_dict:
            ptest = (test_3grams[trigram]**2+1)/(total3grams+len(merged_dict)*(total_test_chars-2))
            ptrain = (train_3grams[trigram]**2+1)/(total3grams+len(merged_dict)*(total_train_chars-2))
            res+=(ptest*math.log(ptest/ptrain))
            
        return res

    def oov_percentage(self):
        len_oovw = len(self.oovw)
        return len_oovw/len(self.train_vocab)*100

    # def perplexity(self):
    #     model = kenlm.Model("corpus/test/fr.foot.test.json")
    #     print(model.score('this is a sentence .', bos = True, eos = True))




    







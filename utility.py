import json
from itertools import zip_longest
from perceptron import Perceptron
import numpy as np
import os
from collections import defaultdict
from collections import OrderedDict
import operator
import _pickle as pickle
import math


"""
renvoie l'alphabet correspondant au set passé en paramètre
"""
def make_alphabet(set_):
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
renvoie le vocabulaire correspondant au set passé en paramètre
"""

def make_vocab(set_):
    rep = dict()
    for sent, _ in set_:
        for w in sent:
            if w in rep:
                rep[w]+=1
            else:
                rep[w]=1
    return rep


"""
renvoie le dictionnaire de bigrammes correspondant au set passé en paramètre
chaque clé correspond à un bigram, et la valeur attachée est le nombre d'occurences de ce bigram
"""

def w_bigrams_dict(set_):
    rep = defaultdict(int)
    for sent, _ in set_: 
        for i in range(1,len(sent)-1):
            bigram = sent[i-1:i+1]
            bigram="{}-{}".format(bigram[0], bigram[1])
            if(bigram in rep):
                rep[bigram]+=1
            else:
                rep[bigram]=1
    return rep

"""
renvoie le vecteur de features correspondant au mot passé en paramètre,
vecteur calculé à l'aide de la phrase, le vocabulaire (supposé être ici composé des N mots les plus fréquents, en accord avec les features décrites en [4])
ainsi qu'un dictionnaire de bigrammes
"""

def sparse_representation(word, sent, vocab, bigram_dict):
    sparse_rep = dict()
    
    sparse_rep["w : {}".format(word)]=1 #word
    
    wid = sent.index(word)
    if(wid>0):    
        sparse_rep["wI-1:{}".format(sent[wid-1])]=1# word at pos i-1
    else:
        sparse_rep["wI-1"]=1
        
    if(wid>1):    
        sparse_rep["wI-2:{}".format(sent[wid-2])]=1# word at pos i-2
    else:
        sparse_rep["wI-2"]=1
        
    if(wid<len(sent)-1):
        sparse_rep["wI+1:{}".format(sent[wid+1])]=1#word at pos i+1
    else:
        sparse_rep["wI+1"]=1
    
    if(wid<len(sent)-2):    
        sparse_rep["wI+2:{}".format(sent[wid+2])]=1#word at pos i+2
    else:
        sparse_rep["wI+2"]=1
        
    for i in range(len(word)):#Suffixes
        sparse_rep["suffixe longueur {} : {}".format(i+1, word[len(word)-(i+1):])] = 1


    if(wid>0):
        wileft = wid - 1
        wleft=sent[wileft]
        if wleft in vocab:
            w_bigram = "{}-{}".format(wleft, vocab[i])
            f_bigram = bigram_dict[w_bigram]
            if(f_bigram>0): 
                f_bigram = 1+math.log(f_bigram)
                f_bigram = round(f_bigram, 3)
                featname="left-bigram:{}, freq : {}".format(w_bigram, f_bigram)
                sparse_rep[featname]=1



    if(wid<len(sent)-1):
        wiright = wid + 1
        wright = sent[wiright]
        if(wright in vocab):
            w_bigram = "{}-{}".format(vocab[i], wright)
            f_bigram = bigram_dict[w_bigram]
            if(f_bigram>0): 
                f_bigram = 1+math.log(f_bigram)
                f_bigram = round(f_bigram, 3)
                featname="right-bigram:{}, freq : {}".format(w_bigram, f_bigram)
                sparse_rep[featname]=1

    return sparse_rep

"""
renvoie le corpus de fichier correspondant au tableau passé en 2nd paramètre (ex: ["foot", "gsd"])
corpusName peux avoir les valeurs suivantes : "test", "train", "dev"
"""

def loadCorpus(corpusName, files):
    set_list = list() 
    for filename in files:
        path = "corpus/{}/fr.{}.{}.json".format(corpusName, filename, corpusName)
        set_list.append(json.load(open(path,"r")))
    set_ = set_list[0]
    for ts in set_list[1:]:
        set_+=ts
    return set_

"""
renvoie le vocabulaire trié par ordre décroissant de fréquence des mots
"""
def sorted_vocab(vocab):
    return sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)

"""
renvoie les x mots les plus fréquents du vocabulaire
"""
def best_x_in_vocab(vocab, x):
    t_list = [w for w in map(list,zip(vocab[:x]))]
    best_x= [w[0] for w in t_list]
    best_x= [w[0] for w in best_x]

    return best_x

"""
renvoie la liste des mots ambigus du set passé en paramètre
"""
def ambiguous_words(set_):
    w_lbl_dict = defaultdict(list)
    ambigous_w_list = list()
    for sent, lbls in set_:
        for w, lbl in zip(sent, lbls):
            if not lbl in w_lbl_dict[w]:
                w_lbl_dict[w].append(lbl)
                
    for w in w_lbl_dict:
        if(len(w_lbl_dict[w])>1):
            ambigous_w_list.append(w)
            
    return ambigous_w_list

"""
renvoie une liste de string composée de la concaténation des mots de chaque liste composant le set
"""
def set_to_string_list(set_):
    rep = list()
    for sent, _ in set_:
        string =""
        for w in sent:
            string+="{} ".format(w)
        rep.append(string.rstrip())
    return rep
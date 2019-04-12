import json
from itertools import zip_longest
from perceptron import Perceptron
from noisinessMeasurer import NoisinessMeasurer
import numpy as np
import os
from collections import defaultdict
from collections import OrderedDict
import operator
import _pickle as pickle
import math
import utility


def precision(p,oovw, ambiguous_w, test_set, vocab, bigram_dict):
    score_global = 0
    score_oovw=0
    score_ambiguous=0

    nw_global=0
    nw_oovw=0
    nw_ambiguous=0

    ns=0
    for sent, labs in test_set:
        ns+=1
        
        for w, lbl in zip(sent, labs):
            features = utility.sparse_representation(w, sent, vocab, bigram_dict)
            pred = p.predict(features)
            nw_global+=1
            if(pred==lbl):
                score_global+=1
                
            if(w in oovw):
                nw_oovw+=1
                if(pred==lbl):
                    score_oovw+=1
                    
            if(w in ambiguous_w):
                nw_ambiguous+=1
                if(pred==lbl):
                    score_ambiguous+=1

            
        if(ns%1000==0):
            print("précision globale époque : ", score_global/nw_global)
            print("précision OOVW époque : ", score_oovw/nw_oovw)
            print("précision sur mots ambigus époque : ", score_ambiguous/nw_ambiguous)


    print("précision globale finale : ", score_global/nw_global)
    print("précision OOVW finale : ", score_oovw/nw_oovw)
    print("précision sur mots ambigus finale : ", score_ambiguous/nw_ambiguous)

    return score_global/nw_global, score_oovw/nw_oovw, score_ambiguous/nw_ambiguous











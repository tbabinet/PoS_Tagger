
"""
Created on Mon Apr  8 18:21:39 2019

@author: tbabi

entraine le perceptron passé en paramètre, selon le nombre d'epochs, le train set, le vocabulaire, et le dictionnaire de bigrammes
"""

import json
from itertools import zip_longest
from perceptron import Perceptron
import numpy as np
import os
from noisinessMeasurer import NoisinessMeasurer 
from collections import defaultdict
from collections import OrderedDict
import operator
import _pickle as pickle
import math
import utility



def train(p, max_epoch, train_set, vocab, bigrams):
    global_score = 0
    global_nw = 0
    ns = 0
    precisions = []

    for epoch in range(max_epoch):
        for sent, labs in train_set:
            ns+=1
            for w, l in zip(sent, labs):
                features = utility.sparse_representation(w, sent, vocab, bigrams )
                pred = p.predict(features)
                p.update(l, pred, features)   
                global_nw+=1
                if(pred==l):
                    global_score+=1
            if(ns%1000==0):
                print("score époque train : ",global_score/global_nw)

        precisions.append(global_score/global_nw)
        ####RESET SCORES
        global_score = 0
        global_nw = 0
        print("epoch : ",epoch)
        
    print("apprentissage terminé")
    print("#######################")
    return precisions




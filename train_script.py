# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:21:39 2019

@author: tbabi
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

# train_files = ["ftb", "gsd", "partut", "pud", "sequoia" ,"spoken"]
# dev_files = ["ftb", "gsd", "partut", "sequoia" ,"spoken"]
# test_files = ["foot","ftb", "gsd","natdis", "partut", "pud", "sequoia" ,"spoken"]


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

# ########TRAIN SET###########
# traindir = [train_files[0]]
# train_set = utility.loadCorpus("train", traindir)
# ###########################################
# ###########################################
# lab_set = set()
# for _, labels in train_set:
#     for label in labels:
#         lab_set.add(label)     
# lab_set = list(lab_set)



# train_vocab = utility.make_vocab(train_set)
# sorted_train_vocab = utility.sorted_vocab(train_vocab)
# best_500 = utility.best_x_in_vocab(sorted_train_vocab, 500)

# train_bigrams = utility.w_bigrams_dict(train_set)

# str_=""
# for filename in traindir:
#     str_+="{}-".format(filename)
# str_=str_[:len(str_)-1]

# filename = "perceptron_with_train_sets_{}.pkl".format(str_)



# p = Perceptron(lab_set)
# train(p, 1, best_500, train_bigrams)

# p.average_weights()

# with open(filename, "wb") as ifile:
#     pickle.dump(p, ifile)


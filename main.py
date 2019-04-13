import json
from itertools import zip_longest
from itertools import combinations
from perceptron import Perceptron
import numpy as np
import os
from noisinessMeasurer import NoisinessMeasurer 
from collections import defaultdict
from collections import OrderedDict
import operator
import _pickle as pickle
import utility
import test_script
import train_script
import csv

"""
train de toutes les combinaisons possibles de perceptron, et tests desdits percepetrons sur tous les test sets possibles
"""

train_files = ["ftb", "gsd", "partut", "pud", "sequoia" ,"spoken"]
test_files = ["foot","ftb", "gsd","natdis", "partut", "pud", "sequoia" ,"spoken"]

for l in range(len(train_files)-4):
        for trainlist in combinations(train_files, l+4):
                print("train list : ",trainlist)
                trainlist = list(trainlist)
                ########TRAIN SET###########
                train_set = utility.loadCorpus("train", trainlist)
                lab_set = set()
                for _, labels in train_set:
                        for label in labels:
                                lab_set.add(label)

                lab_set = list(lab_set)
                train_vocab = utility.make_vocab(train_set)
                sorted_train_vocab = utility.sorted_vocab(train_vocab)
                best_500 = utility.best_x_in_vocab(sorted_train_vocab, 500)

                train_bigrams = utility.w_bigrams_dict(train_set)
                p=Perceptron(lab_set)
                train_precisions = train_script.train(p, 10, train_set, best_500, train_bigrams)

                traincorpus="_".join(trainlist)

                for i in range(len(test_files)):
                        for testlist in combinations(test_files, i+1):
                                
                                print("test list : ",testlist)
                                testlist = list(testlist)
                                ########TEST SET#########
                                test_set = utility.loadCorpus("test", testlist)

                                nm = NoisinessMeasurer(train_set, test_set)
                                oovw_list = nm.oovw
                                KLD = nm.KLD
                                #perplexity = nm.perplexity(testlist)
                                ambigous_w_list = utility.ambiguous_words(train_set)

                                testcorpus="_".join(testlist)
                                
                                precision_globale, precision_oov, precision_ambiguous = test_script.precision(p,oovw_list, ambigous_w_list, test_set, best_500, train_bigrams)
                                row = [precision_globale, precision_oov, precision_ambiguous, nm.oov_percentage(), nm.KLD]
                                print(len(train_precisions))
                                for precision in train_precisions[::-1]:
                                        row.insert(0, precision)


                                row_rounded = [round(value, 10) for value in row]
                                row_rounded.insert(0, testcorpus)
                                row_rounded.insert(0, traincorpus)
                                print(row_rounded)
                                with open('results_tour.csv', 'a') as datafile:
                                        writer = csv.writer(datafile)
                                        writer.writerow(row_rounded)
                                

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
import utility
import test_script
import train_script
import csv

train_files = ["ftb", "gsd", "partut", "pud", "sequoia" ,"spoken"]
test_files = ["foot","ftb", "gsd","natdis", "partut", "pud", "sequoia" ,"spoken"]

ftb = "ftb"
gsd="gsd"
partut="partut"
pud="pud"
sequoia="sequoia"
spoken="spoken"
foot="foot"
natdis="natdis"

traindirs = [partut, pud, sequoia, spoken]

testdirs = [
        [ftb], 
        [gsd], 
        [foot], 
        [ftb, gsd], 
        [ftb, foot], 
        [foot, gsd],
        [foot, gsd, ftb]
         ]




         
########TRAIN SET###########
train_set = utility.loadCorpus("train", traindirs)
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

traincorpus=""
for filename in traindirs:
        traincorpus+="{}-".format(filename)
traincorpus=traincorpus[:len(traincorpus)-1]

for testlist in testdirs:

        print("train list : ",traindirs)
        print("test list : ",testlist)
        
        ########TEST SET#########
        test_set = utility.loadCorpus("test", testlist)

        nm = NoisinessMeasurer(train_set, test_set)
        oovw_list = nm.oovw
        KLD = nm.KLD
        ambigous_w_list = utility.ambiguous_words(train_set)

        testcorpus=""
        for filename in testlist:
                testcorpus+="{}-".format(filename)
        testcorpus=testcorpus[:len(testcorpus)-1]

        
        
        precision_globale, precision_oov, precision_ambiguous = test_script.precision(p,oovw_list, ambigous_w_list, test_set, best_500, train_bigrams)
        row = [precision_globale, precision_oov, precision_ambiguous, nm.oov_percentage(), nm.KLD ]
        print(len(train_precisions))
        for precision in train_precisions[::-1]:
                row.insert(0, precision)


        row_rounded = [round(value, 10) for value in row]
        row_rounded.insert(0, testcorpus)
        row_rounded.insert(0, traincorpus)
        print(row_rounded)
        with open('data.csv', 'a') as datafile:
                writer = csv.writer(datafile)
                writer.writerow(row_rounded)
        datafile.close()

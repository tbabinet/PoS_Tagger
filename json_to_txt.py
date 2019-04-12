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
import utility
import sys
from itertools import combinations


test_files = ["foot","ftb", "gsd","natdis", "partut", "pud", "sequoia" ,"spoken"]


def json_to_txt(fileList):
    set_ = utility.loadCorpus("test", fileList)
    print("load ok")
    txtList = utility.set_to_string_list(set_)
    print("set to list ok")
    txtfile=""
    for filename in subset:
            txtfile+="{}_".format(filename)
    txtfile = txtfile[:(len(txtfile)-1)]
    print(txtfile)
    with open ("txt/{}.txt".format(txtfile), "a", encoding="utf-8") as txtfile:
        print("file opened")
        for line in txtList:
            txtfile.write(line)
            txtfile.write("\n")
    print("done")





for l in range(len(test_files)):
    for subset in combinations(test_files, l+1):
        subset = list(subset)
        json_to_txt(subset)
        
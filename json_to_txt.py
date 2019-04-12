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



def json_to_txt(jsonFile):
    set_ = utility.loadCorpus("test", [jsonFile])
    print("load ok")
    txtList = utility.set_to_string_list(set_)
    print("set to list ok")
    with open ("txt/{}.txt".format(jsonFile), "a", encoding="utf-8") as txtfile:
        print("file opened")
        for line in txtList:
            txtfile.write(line)
            txtfile.write("\n")
    print("done")


corpus = sys.argv[1]
json_to_txt(corpus)
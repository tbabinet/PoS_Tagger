import subprocess
from itertools import combinations

"""
script créant l'ensembles des fichiers .arpa nécessaires au calcul de la perplxité
"""

test_files = ["foot","ftb", "gsd","natdis", "partut", "pud", "sequoia" ,"spoken"]

for l in range(len(test_files)+1):
    for subset in combinations(test_files, l):
        txtfile = ""
        for filename in subset:
            txtfile+="{}_".format(filename)
        txtfile = txtfile[:(len(txtfile)-1)]
        infile = "<txt/{}.txt".format(txtfile)
        outfile = ">arpa/{}.arpa".format(txtfile)
        cmd = "../kenlm/build/bin/lmplz -o 5 {} {} --discount_fallback".format(infile, outfile)
        subprocess.call(cmd, shell=True)

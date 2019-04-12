import kenlm
import sys
import utility

def perplexity(mod, seq):
    print(model.perplexity(seq))




arpaFile = sys.argv[1]
model = kenlm.Model("arpa/{}.arpa".format(arpaFile))
corpus = utility.loadCorpus("test", [arpaFile])
listcorpus = utility.set_to_string_list(corpus)
corpus_str =""
for sent in listcorpus:
    corpus_str+=("{} ".format(sent))

corpus_str = corpus_str.rstrip()
print(corpus_str)
perplexity(model, corpus_str)

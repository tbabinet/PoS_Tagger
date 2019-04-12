import numpy as np
from collections import defaultdict
from itertools import zip_longest

class Perceptron:
    def __init__(self, labels):
        self.labels = labels
        # Each feature gets its own weight vector, with one weight for
        # each possible label
        self.weights = defaultdict(lambda: defaultdict(float))
        # The accumulated values of the weight vector at the t-th
        # iteration: sum_{i=1}^{n - 1} w_i
        #
        # The current value (w_t) is not yet added. The key of this
        # dictionary is a pair (feature, label)
        self._accum = defaultdict(int)
        # The last time the feature was changed, for the averaging.
        self._last_update = defaultdict(int)
        # Number of examples seen
        self.n_updates = 0

    
    def predict(self, features):
    #Dot-product the features and current weights and return the best class.
        
        scores = self.score(features, self.labels)
        maxScore = max(scores)
        idLab=scores.index(maxScore)   
        return self.labels[idLab]

    
    def score(self, features, labels=None):
        
        """
        Parameters
        ----------

        - features, an iterable
             a sequence of binary features. Each feature must be
             hashable. WARNING: the `value' of the feature is always
             assumed to be 1.
        - labels, a subset of self.labels
             if not None, the score is computed only for these labels
        """
        scores = []
        labs = self.labels
        if labels:
            labs = labels

        for label in labs:
            score = sum(self.weights[f][label] for f in features)
            scores.append(score)
        return scores
    

    def update(self, truth, guess, features):
        def upd_feat(label, feature, v):
            param = (label, feature)
            self._accum[param] += (self.n_updates - self._last_update[param]) * self.weights[feature][label]
            self._last_update[param] = self.n_updates
            self.weights[feature][label] += v
            #print(self.weights[feature])
        self.n_updates += 1

        if truth == guess:
            return

        for f in features:
            upd_feat(truth, f, 1.0)
            upd_feat(guess, f, -1.0)

    def average_weights(self):
        """
        Average weights of the perceptron

        Training can no longer be resumed.
        """
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for label, w in weights.items():
                param = (label, feat)
                # Be careful not to add 1 to take into account the
                # last weight vector (without increasing the number of
                # iterations in the averaging)
                total = self._accum[param] + \
                    (self.n_updates + 1 - self._last_update[param]) * w
                averaged = round(total / self.n_updates, 3)
                if averaged:
                    new_feat_weights[label] = averaged
            self.weights[feat] = new_feat_weights

    def __getstate__(self):
        """
        Serialization of a perceptron

        We are only serializing the weight vector as a dictionnary
        because defaultdict with lambda can not be serialized.
        """
        # should we also serialize the other attributes to allow
        # learning to continue?
        return {"weights": {k: v for k, v in self.weights.items()},
                "labels": self.labels
                }

    def __setstate__(self, data):
        """
        De-serialization of a perceptron
        """

        print(data["weights"])
        w_dict = {}
        self.weights = defaultdict(lambda: defaultdict(float), data["weights"])
        self.labels = data["labels"]
        # ensure we are no longer able to continue training
        self._accum = None
        self._last_update = None



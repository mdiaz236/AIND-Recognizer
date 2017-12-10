import warnings
from asl_data import SinglesData
import numpy as np

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for i, Xlengths in test_set.get_all_Xlengths().items():
        X, lengths = Xlengths
        prob_dict = {}
        best_guess = None
        best_prob = -np.inf
        for word, model in models.items():
            try:
                logL = model.score(X, lengths)
                if logL > best_prob:
                    best_prob = logL
                    best_guess = word
                prob_dict[word] = logL
            except (AttributeError, ValueError):
                pass
        probabilities.append(prob_dict)
        guesses.append(best_guess)
    return (probabilities, guesses)

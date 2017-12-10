import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def run_bic_model(self, num_states):
        try:
            model = self.base_model(num_states)
            logL = model.score(self.X, self.lengths)
            N = len(self.X)
            p = num_states ** 2 + 2 * model.n_features * num_states - 1

            bic = p * np.log(N) - 2 * logL
            return (model, bic)
        except (AttributeError, ValueError) as e:
            return None

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        bic_models = map(lambda x: self.run_bic_model(x),
                         range(self.min_n_components, self.max_n_components + 1))
        valid_models = [x for x in bic_models if x is not None]

        if len(valid_models) > 0:
            best_model = sorted(valid_models, key=lambda x: x[1])[0]
            return best_model[0]
        else:
            return None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def score_word(self, num_states):
        model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        logL = model.score(self.X, self.lengths)
        return (model, logL)

    def score_other_word(self, model, word):
        X, lengths = self.hwords[word]
        logL = model.score(X, lengths)
        return logL


    def run_dic_model(self, num_states):
        try:
            alpha = 1.
            (model, word_logL) = self.score_word(num_states)
            other_words = filter(lambda word: word != self.this_word,
                                 self.words.keys())
            other_logL = list(map(lambda x: self.score_other_word(model, x),
                             other_words))
            other_logL_mean = np.nanmean(other_logL)
            dic = word_logL - (alpha * other_logL_mean)
            return (model, dic, num_states)
        except (ValueError, AttributeError) as e:
            return None

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        dic_models = map(lambda x: self.run_dic_model(x),
                         range(self.min_n_components, self.max_n_components + 1))
        valid_models = [x for x in dic_models if x is not None]

        if len(valid_models) > 0:
            best_model = sorted(valid_models, key=lambda x: -x[1])[0]
            return best_model[0]
        else:
            return None

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def run_model(self, train, train_lengths, test, test_lengths, num_states):
        '''
        return log likelyhood
        '''
        model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                          random_state=self.random_state, verbose=False).fit(train, train_lengths)
        logL = model.score(test, test_lengths)
        return logL


    def run_cv_model(self, sequences, num_states):
        try:
            split_method = KFold()
            seq = np.array(sequences)
            all_logL = []
            for train_i, test_i in split_method.split(seq):
                train, train_lengths = combine_sequences(train_i, seq)
                test, test_lengths = combine_sequences(test_i, seq)
                logL = self.run_model(train, train_lengths, test, test_lengths, num_states)
                all_logL.append(logL)
            return (num_states, np.mean(all_logL))
        except (AttributeError, ValueError):
            return (num_states, np.nan)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        results = list(map(lambda x: self.run_cv_model(self.sequences, x),
                           range(self.min_n_components, self.max_n_components + 1)))
        best_states = sorted(results, key=lambda x: -x[1])[0][0]
        best_model = self.base_model(best_states)
        return best_model

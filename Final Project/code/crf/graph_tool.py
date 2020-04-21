from collections import Counter
import numpy as np


class PMI:
    def __init__(self, ngrams, features_list):
        self.ngrams_counter = Counter(ngrams)
        self.sum_ngrams = len(ngrams)

        feature_dict = {}
        ngrams_feature_dict = {}

        for ngram, features in zip(ngrams, features_list):
            for feature_name, feature in features.items():
                if feature_name not in feature_dict:
                    feature_dict[feature_name] = []
                if feature_name not in ngrams_feature_dict:
                    ngrams_feature_dict[feature_name] = []

                # dict for 9 features
                feature_dict[feature_name].append(feature)
                # dict for 9 features with (ngram, feature) pair as elements
                ngrams_feature_dict[feature_name].append((ngram, feature))

        self.feature_counters = {}
        self.ngrams_feature_counters = {}

        for feature_name, feature in feature_dict.items():
            self.feature_counters[feature_name] = Counter(feature)

        for feature_name, ngram_feature in ngrams_feature_dict.items():
            self.ngrams_feature_counters[feature_name] = Counter(ngram_feature)

    def pmi(self, ngram, feature, feature_name):
        count_ngram = self.ngrams_counter[ngram] if self.ngrams_counter[ngram] is not None else 0

        count_feature = self.feature_counters[feature_name][feature] if self.feature_counters[feature_name][
                                                                          feature] is not None else 0

        count_ngram_feature = self.ngrams_feature_counters[feature_name][(ngram, feature)] if \
            self.ngrams_feature_counters[feature_name][(ngram, feature)] is not None else 0

        if count_ngram == 0 or count_feature == 0 or count_ngram_feature == 0:
            score = 0
        else:
            # sum_ngrams = sum_count_feature, so there is a (* self.sum_ngrams)
            score = np.log((count_ngram_feature * self.sum_ngrams) / (count_ngram * count_feature))

        return score

    def pmi_vector(self, ngram, features_dict):
        return np.array(
            [self.pmi(ngram, feature, feature_name) for feature_name, feature_set in features_dict.items() for feature
             in feature_set])

from helper import *
from preprocess import preprocess_label, preprocess_unlabel
from graph_tool import *
import argparse
import sklearn_crfsuite
from collections import Counter
import timeit


def read_data(file):

    with open(file) as f:
        raw_data = f.readlines()

    return raw_data


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_file", default='../data/labeled_train')
    parser.add_argument("--unlabeled_file", default='../data/unlabeled_train')
    args = parser.parse_args()

    # load dataset
    label_data = read_data(args.labeled_file)
    unlabel_data = read_data(args.unlabeled_file)

    label_data = preprocess_label(label_data)
    unlabel_data = preprocess_unlabel(unlabel_data)

    all_data = label_data + unlabel_data
    # extract features
    label_feature = [sent2features(s, type=True) for s in label_data]
    label_target = [sent2labels(s) for s in label_data]

    print('finish: loading features')
    '''
    # compute CRF
    crf = sklearn_crfsuite.CRF(
        algorithm='l2sgd',
        c2=0.01,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(label_feature, label_target)
    '''

    # create graph with labeled and unlabeled data
    ngrams_list = []
    for sent in all_data:
        ngrams = sent2trigrams(sent)
        ngrams_list.extend(ngrams)

    print('finish: get trigrams')

    graph_features_list = []
    for sent in all_data:
        features = sent2graphfeatures(sent)
        graph_features_list.extend(features)

    print('finish: get graph features')

    # compute pmi_vectors
    pmi = PMI(ngrams_list, graph_features_list)
    features_dict = features2dict(graph_features_list)

    start = timeit.default_timer()
    pmi_vectors = np.array([pmi.pmi_vector(ngram, features_dict) for ngram in ngrams_list[0]])
    end = timeit.default_timer()

    print('finish: construct pmi vectors', pmi_vectors.size)
    print('running cost: ', end - start)


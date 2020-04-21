from helper import *
from preprocess import preprocess_label, preprocess_unlabel
import argparse
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from collections import Counter
import timeit


def read_data(file):

    with open(file) as f:
        raw_data = f.readlines()

    return raw_data


if __name__=='__main__':
    start = timeit.default_timer()

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
    end = timeit.default_timer()
    print('running cost: ', end - start)


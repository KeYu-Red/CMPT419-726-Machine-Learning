from helper import *
from preprocess import preprocess_label, preprocess_unlabel
import argparse


def read_data(file):

    with open(file) as f:
        raw_data = f.readlines()

    return raw_data


def SentsToFeatures(data):
    ret = []
    for sent in data:
        for i in range(len(sent)):

            feature = {}

            ret.append(sent)

    return ret


def SentsToTarget(data):
    ret = []
    for sent in data:
        for word in sent:
            ret.append(word[3])

    return ret


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_file", default='labeled_train')
    parser.add_argument("--unlabeled_file", default='unlabeled_train')
    args = parser.parse_args()

    # load dataset
    label_data = read_data(args.labeled_file)
    unlabel_data = read_data(args.unlabeled_file)

    labeled = preprocess_label(label_data)
    #unlabeled = preprocess_unlabel(unlabel_data)

    # extract feature from words
    label_feature = SentsToFeatures(label_data)
    label_target = SentsToTarget(label_data)

from helper import *
from preprocess import preprocess_label, preprocess_unlabel
import argparse
import operator
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from collections import Counter
import timeit
import random
import matplotlib.pyplot as plt


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
    label_feature = [sent2features(s, type=False) for s in label_data]
    label_target = [sent2labels(s) for s in label_data]
    label_feature_train, label_feature_test, label_target_train, label_target_test = train_test_split(
                                label_feature, label_target, train_size=0.8, test_size=0.2, random_state=21)
    unlabel_feature = [sent2features(s, type=False) for s in unlabel_data]

    print('finish: loading features')


    # compute CRF
    crf = sklearn_crfsuite.CRF(
        algorithm='l2sgd',
        c2=0.01,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(label_feature_train, label_target_train)


    accuracy_list = []
    dataAdd_list = []


    end = timeit.default_timer()
    test_predict = crf.predict(label_feature_test)
    count_the_same = 0
    for t in range(len(test_predict)):
        if(test_predict[t]==label_target_test[t]):
            count_the_same += 1
    #####-----End of For-----####
    accuracy_list.append(100.0*count_the_same/len(test_predict))
    print('finish: training initial classifier c0 accuracy %.6f%%' % (100.0*count_the_same/len(test_predict)))
    print('running cost: ', end - start)


    # retrain
    start = timeit.default_timer()
    batch_size = 50
    K = 50 # int(len(unlabel_feature)/batch_size)
    T_score = 0.98
    last_score = 0
    for i in range(K):
        old_crf = crf
        high_confi_dic = {}
        new_features = []
        new_predict = []
        retrain_predict = crf.predict_marginals(unlabel_feature[i*batch_size : (i+1)*batch_size])
        data_added = 0
        sentence_added = 0
        sentence_num = 0
        for sentence in retrain_predict:
            token_num = 0
            for token in sentence:
                word = unlabel_feature[i*batch_size + sentence_num][token_num]['word.lower()']
                tag, max_score = max(token.items(), key=operator.itemgetter(1))
                if max_score > T_score and tag != 'O':
                    high_confi_dic[word] = tag
                token_num += 1
            #####-----End of For-----####
            sentence_num += 1
        #####-----End of For-----####
        sentence_num = 0
        for sentence in retrain_predict:
            token_num = 0
            new_features_2 = []
            new_predict_2 = []
            for token in sentence:
                word = unlabel_feature[i*batch_size + sentence_num][token_num]['word.lower()']
                tag, max_score = max(token.items(), key=operator.itemgetter(1))
                if max_score <= T_score:
                    if word in high_confi_dic.keys():
                        new_features_2.append(unlabel_feature[i*batch_size + sentence_num][token_num])
                        new_predict_2.append(high_confi_dic[word])
                        data_added += 1
                elif tag != 'O':
                    new_features_2.append(unlabel_feature[i*batch_size + sentence_num][token_num])
                    new_predict_2.append(tag)
                    data_added += 1
                #####-----End of If-----####
                token_num += 1
            #####-----End of For-----####
            sentence_num += 1
            if data_added > 0:
                new_features.append(new_features_2)
                new_predict.append(new_predict_2)
                sentence_added += 1
        #####-----End of For-----####

        label_feature_train += new_features
        label_target_train += new_predict
        crf.fit(label_feature_train, label_target_train)
        test_predict = crf.predict(label_feature_test)
        count_the_same = 0
        for t in range(len(test_predict)):
            if(test_predict[t]==label_target_test[t]):
                count_the_same += 1
        #####-----End of For-----####
        current_score = 100.0*count_the_same/len(test_predict)
        current_score_str = '%.6f'%current_score
        print(f'finish: training c{i+1} total number {K} accuracy {current_score_str}% dataAdd {data_added}')
        accuracy_list.append(current_score)
        dataAdd_list.append(data_added)
        # if last_score > current_score:
        #     if sentence_added > 0:
        #         label_feature_train = label_feature_train[:-sentence_added]
        #         label_target_train = label_target_train[:-sentence_added]
        #     crf = old_crf
        #     print('Abort')
        # else:
        #     last_score = current_score
    #####-----End of For-----####


    end = timeit.default_timer()
    print('running cost: ', end - start)

    plt.figure(1)
    plt.plot(range(len(accuracy_list)), accuracy_list, linewidth=1)
    plt.xlabel('Episode Number')
    plt.ylabel('Accuracy')
    plt.show()

    plt.figure(2)
    plt.plot(range(1, 1+len(dataAdd_list)), dataAdd_list, linewidth=1)
    plt.xlabel('Episode Number')
    plt.ylabel('Number of Tokens Added')
    plt.show()


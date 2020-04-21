# convert all graph features to a dictionary
def features2dict(features_list):
    dict = {}
    for features in features_list:
        for feature_name, feature in features.items():
            if feature_name not in dict:
                dict[feature_name] = set()
            dict[feature_name].add(feature)

    return dict


def has_suffix(word):
    suffixes = ["ed", "ing"]
    for suffix in suffixes:
        if word.endswith(suffix):
            return True
    return False


def word2graphfeatures(sent, i):
    # trigram, 5-nearest neighbor, i start from 1, first trigram
    words = [ word[0] for word in sent[i-2:i+3]]

    features = {
        'trigram+context': ' '.join(words),
        'trigram': ' '.join(words[1:4]),
        'left_context': ' '.join(words[0:2]),
        'right_context': ' '.join(words[3:5]),
        'center_word': words[2],
        'trigram-centerword': words[1] +' '+ words[3],
        'left_word-right_context': words[1] + ' ' + ' '.join(words[3:5]),
        'left_context-right_word': ' '.join(words[0:2]) + ' ' + words[3],
        'suffix': has_suffix(words[2])
    }
    return features


def sent2graphfeatures(sent):
    sent = [('<BOS>', '<BOS>', '', 'O')] + sent + [('<EOS>', '<EOS>', '', 'O')]

    graph_features = []
    for i in range(len(sent) - 4):
        feature = word2graphfeatures(sent, i+2)
        graph_features.append((feature))

    return graph_features


def sent2trigrams(sent):
    trigrams = []
    for i in range(len(sent)-2):
        trigram = [word[0] for word in (sent[i:i+3])]
        trigrams.append(' '.join(trigram))

    return trigrams


# three methods below are adopted and modified from sklearn-crfsuite documentation
def word2features(sent, i, type):
    word = sent[i][0]
    postag = sent[i][1]

    if type:
        # features used by paper
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[0:3]': word[0:3],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
        }
        if i > 0:
            features['BOS'] = False
        else:
            features['BOS'] = True

        if i < len(sent) - 1:
            features['EOS'] = False
        else:
            features['EOS'] = True
    else:
        # features from documentation
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

    return features


def sent2features(sent, type):
    return [word2features(sent, i, type) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, chunk, label in sent]



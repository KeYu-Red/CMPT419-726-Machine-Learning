# function to preprocess data
def preprocess_label(data):
    sent_set = []
    sentence = []
    for line in data:
        if line.split():
            sentence.append(tuple(line.split()))
        else:
            sent_set.append(sentence)
            sentence = []

    return sent_set

# function to get unlabelled data
def get_unlabelled_data(labelled_data):
    import copy
    unlabelled_data = copy.deepcopy(labelled_data)
    for sentence in unlabelled_data:
        for word in sentence:
            j = sentence.index(word)
            word = list(word)
            word[3] = 'O'
            sentence[j] = word
    return unlabelled_data

# function to get training data
def get_train_data(unlabelled_data, spl_dl, ctx_dl, choice):
    import copy
    train_data = copy.deepcopy(unlabelled_data)

    if choice == 'contextual':
        for sentence in train_data:
            for word in sentence:
                if word[0] in spl_dl:
                    word[3] = spl_dl[word[0]]

    if choice == 'spelling':
        for sentence in train_data:
            len_sentence = len(sentence)
            for word in sentence:
                i = sentence.index(word)
                if (sentence[i][0]+'-p2') in ctx_dl:
                    if i+2 < len_sentence:
                        sentence[i+2][3] = ctx_dl[sentence[i][0]+'-p2']
                if (sentence[i][0]+'-p1') in ctx_dl:
                    if i+1 < len_sentence:
                        sentence[i+1][3] = ctx_dl[sentence[i][0]+'-p1']
                if (sentence[i][0]+'-r1') in ctx_dl:
                    if i-1 >= 0:
                        sentence[i-1][3] = ctx_dl[sentence[i][0]+'-r1']
                if (sentence[i][0]+'-r2') in ctx_dl:
                    if i-2 >= 0:
                        sentence[i-2][3] = ctx_dl[sentence[i][0]+'-r2']

    return train_data

# function to get the prediction accuracy
def test(ctx_dl, spl_dl, test_data):
    correct = 0
    total = 0
    for sentence in test_data:
        len_sentence = len(sentence)
        for word in sentence:
            i = sentence.index(word)
            spl_feature = word[0]
            prediction_list = []

            if spl_feature in spl_dl:
                spl_prediction = spl_dl[spl_feature]
                prediction_list.append(spl_prediction)

            if i-2 >= 0:
                ctx_p2_feature = sentence[i-2][0]+'-p2'
                if ctx_p2_feature in ctx_dl:
                    ctx_p2_prediction = ctx_dl[ctx_p2_feature]
                    prediction_list.append(ctx_p2_prediction)

            if i-1 >= 0:
                ctx_p1_feature = sentence[i-1][0]+'-p1'
                if ctx_p1_feature in ctx_dl:
                    ctx_p1_prediction = ctx_dl[ctx_p1_feature]
                    prediction_list.append(ctx_p1_prediction)

            if i+1 < len_sentence:
                ctx_r1_feature = sentence[i+1][0]+'-r1'
                if ctx_r1_feature in ctx_dl:
                    ctx_r1_prediction = ctx_dl[ctx_r1_feature]
                    prediction_list.append(ctx_r1_prediction)

            if i+2 < len_sentence:
                ctx_r2_feature = sentence[i+2][0]+'-r2'
                if ctx_r2_feature in ctx_dl:
                    ctx_r2_prediction = ctx_dl[ctx_r2_feature]
                    prediction_list.append(ctx_r2_prediction)

            if len(prediction_list) > 1:
                from collections import Counter
                c = Counter(prediction_list)
                prediction, count = c.most_common()[0]

                if prediction == word[3]:
                    correct += 1
            if word[3]!='O' and word[3]!='I-MISC':
                total += 1
    return correct/total


# load the file
file = "labeled_train"
with open(file) as f:
    raw_data = f.readlines()
labelled_data = preprocess_label(raw_data)

# get the entities in the dataset
labels = ['I-PER', 'I-ORG', 'I-LOC']
entities_sentence_idx = []
entities_word_idx = []
entities_idx = []
for sentence in labelled_data:
    for word in sentence:
        if word[3] in labels:
            entities_sentence_idx.append(labelled_data.index(sentence))
            entities_word_idx.append(sentence.index(word))
            entities_idx.append(
                [labelled_data.index(sentence), sentence.index(word)])

# get the seeds
seed_idx = sample(entities_idx, round(len(entities_idx)/40))

# initialize the DLs
spl_dl = {}
ctx_dl = {}
for seed_iter in seed_idx:
    word = labelled_data[seed_iter[0]][seed_iter[1]]
    spl_feature = word[0]
    label = word[3]
    spl_dl[spl_feature] = label

# get the unlabelled data
unlabelled_data = get_unlabelled_data(labelled_data)

# main loop
n = 5
threshold = 0.95
while n <= 3000:
    n += 5
    h_per = {}
    h_org = {}
    h_loc = {}
    count_per = {}
    count_org = {}
    count_loc = {}
    features = []
    train_data = get_train_data(unlabelled_data, spl_dl, ctx_dl, 'contextual')

    # step1: spelling features -> contextual features
    for sentence in train_data:
        i = 0
        len_sentence = len(sentence)
        for word in sentence:
            prior_2 = []
            prior_1 = []
            rear_1 = []
            rear_2 = []
            if i > 0:
                prior_1 = sentence[i-1]
            if i > 1:
                prior_2 = sentence[i-2]
            if i+1 < len_sentence:
                rear_1 = sentence[i+1]
            if i+2 < len_sentence:
                rear_2 = sentence[i+2]

            # calculate contextual features of objects libeled with "I-PER"
            if word[3] == 'I-PER':
                if len(prior_2) != 0:
                    #print(prior_2)
                    feature = prior_2[0]+'-p2'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_per:
                            count_per[feature] += 1
                        else:
                            count_per[feature] = 1
                if len(prior_1) != 0:
                    feature = prior_1[0]+'-p1'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_per:
                            count_per[feature] += 1
                        else:
                            count_per[feature] = 1
                if len(rear_1) != 0:
                    feature = rear_1[0]+'-r1'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_per:
                            count_per[feature] += 1
                        else:
                            count_per[feature] = 1
                if len(rear_2) != 0:
                    feature = rear_2[0]+'-r2'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_per:
                            count_per[feature] += 1
                        else:
                            count_per[feature] = 1

            # calculate contextual features of objects libeled with "I-ORG"
            if word[3] == 'I-ORG':
                if len(prior_2) != 0:
                    feature = prior_2[0]+'-p2'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_org:
                            count_org[feature] += 1
                        else:
                            count_org[feature] = 1
                if len(prior_1) != 0:
                    feature = prior_1[0]+'-p1'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_org:
                            count_org[feature] += 1
                        else:
                            count_org[feature] = 1
                if len(rear_1) != 0:
                    feature = rear_1[0]+'-r1'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_org:
                            count_org[feature] += 1
                        else:
                            count_org[feature] = 1
                if len(rear_2) != 0:
                    feature = rear_2[0]+'-r2'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_org:
                            count_org[feature] += 1
                        else:
                            count_org[feature] = 1

            # calculate contextual features of objects libeled with "I-LOC"
            if word[3] == 'I-LOC':
                if len(prior_2) != 0:
                    feature = prior_2[0]+'-p2'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_loc:
                            count_loc[feature] += 1
                        else:
                            count_loc[feature] = 1
                if len(prior_1) != 0:
                    feature = prior_1[0]+'-p1'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_loc:
                            count_loc[feature] += 1
                        else:
                            count_loc[feature] = 1
                if len(rear_1) != 0:
                    feature = rear_1[0]+'-r1'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_loc:
                            count_loc[feature] += 1
                        else:
                            count_loc[feature] = 1
                if len(rear_2) != 0:
                    feature = rear_2[0]+'-r2'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_loc:
                            count_loc[feature] += 1
                        else:
                            count_loc[feature] = 1
                if len(rear_1) != 0:
                    feature = rear_1[0]+'-r1'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_loc:
                            count_loc[feature] += 1
                        else:
                            count_loc[feature] = 1
                if len(rear_2) != 0:
                    feature = rear_2[0]+'-r2'
                    if feature not in ctx_dl:
                        features.append(feature)
                        if feature in count_loc:
                            count_loc[feature] += 1
                        else:
                            count_loc[feature] = 1
            i += 1

    # remove duplicates from features
    features = list(dict.fromkeys(features))
    # compute h
    for feature in features:
        if feature not in count_per:
            count_per[feature] = 0
        if feature not in count_org:
            count_org[feature] = 0
        if feature not in count_loc:
            count_loc[feature] = 0
        h_sum = count_per[feature] + count_org[feature] + count_loc[feature]
        h_per[feature] = count_per[feature]/h_sum
        h_org[feature] = count_org[feature]/h_sum
        h_loc[feature] = count_loc[feature]/h_sum
    features_per = [k for k, v in h_per.items() if v >= threshold]
    features_org = [k for k, v in h_org.items() if v >= threshold]
    features_loc = [k for k, v in h_loc.items() if v >= threshold]

    # select top n features and add them to DL
    count_per = dict((k, v) for k, v in count_per.items() if k in features_per)
    count_org = dict((k, v) for k, v in count_org.items() if k in features_org)
    count_loc = dict((k, v) for k, v in count_loc.items() if k in features_loc)
    from collections import Counter
    k = Counter(count_per) 
    dl_per = k.most_common(n)
    dl_per = dict(dl_per)
    
    dl_per = dict((k,v) for k,v in dl_per.items() if v > 3)
    dl_per.update((x,'I-PER') for x in dl_per)
    
    k = Counter(count_org) 
    dl_org = k.most_common(n)
    dl_org = dict(dl_org)
    
    dl_org = dict((k,v) for k,v in dl_org.items() if v > 3)
    dl_org.update((x,'I-ORG') for x in dl_org)
    
    k = Counter(count_loc) 
    dl_loc = k.most_common(n)
    dl_loc = dict(dl_loc)
    
    dl_loc = dict((k,v) for k,v in dl_loc.items() if v > 3)
    dl_loc.update((x,'I-LOC') for x in dl_loc)

    # add to DL
    ctx_dl.update(dl_per)
    ctx_dl.update(dl_org)
    ctx_dl.update(dl_loc)
    print('step1 done')

    # step2: contextual features -> spelling features
    h_per = {}
    h_org = {}
    h_loc = {}
    count_per = {}
    count_org = {}
    count_loc = {}
    features = []
    train_data = get_train_data(unlabelled_data, spl_dl, ctx_dl, 'spelling')
    for sentence in train_data:
        for word in sentence:

            # calculate contextual features of objects libeled with "I-PER"
            if word[3] == 'I-PER':
                feature = word[0]
                if feature not in spl_dl:
                    features.append(feature)
                    if feature in count_per:
                        count_per[feature] += 1
                    else:
                        count_per[feature] = 1

            # calculate contextual features of objects libeled with "I-ORG"
            if word[3] == 'I-ORG':
                feature = word[0]
                if feature not in spl_dl:
                    features.append(feature)
                    if feature in count_org:
                        count_org[feature] += 1
                    else:
                        count_org[feature] = 1

            # calculate contextual features of objects libeled with "I-LOC"
            if word[3] == 'I-LOC':
                feature = word[0]
                if feature not in spl_dl:
                    features.append(feature)
                    if feature in count_loc:
                        count_loc[feature] += 1
                    else:
                        count_loc[feature] = 1

    # remove duplicates from features
    features = list(dict.fromkeys(features))
    # compute h
    for feature in features:
        if feature not in count_per:
            count_per[feature] = 0
        if feature not in count_org:
            count_org[feature] = 0
        if feature not in count_loc:
            count_loc[feature] = 0
        h_sum = count_per[feature] + count_org[feature] + count_loc[feature]
        h_per[feature] = count_per[feature]/h_sum
        h_org[feature] = count_org[feature]/h_sum
        h_loc[feature] = count_loc[feature]/h_sum
    features_per = [k for k, v in h_per.items() if v >= threshold]
    features_org = [k for k, v in h_org.items() if v >= threshold]
    features_loc = [k for k, v in h_loc.items() if v >= threshold]

    # select top n features and add them to DL
    count_per = dict((k, v) for k, v in count_per.items() if k in features_per)
    count_org = dict((k, v) for k, v in count_org.items() if k in features_org)
    count_loc = dict((k, v) for k, v in count_loc.items() if k in features_loc)
    from collections import Counter
    k = Counter(count_per) 
    dl_per = k.most_common(n)
    dl_per = dict(dl_per)
    
    dl_per = dict((k,v) for k,v in dl_per.items() if v > 3)
    dl_per.update((x,'I-PER') for x in dl_per)
    
    k = Counter(count_org) 
    dl_org = k.most_common(n)
    dl_org = dict(dl_org)
    
    dl_org = dict((k,v) for k,v in dl_org.items() if v > 3)
    dl_org.update((x,'I-ORG') for x in dl_org)
    
    k = Counter(count_loc) 
    dl_loc = k.most_common(n)
    dl_loc = dict(dl_loc)
    
    dl_loc = dict((k,v) for k,v in dl_loc.items() if v > 3)
    dl_loc.update((x,'I-LOC') for x in dl_loc)

    # add to DL
    spl_dl.update(dl_per)
    spl_dl.update(dl_org)
    spl_dl.update(dl_loc)
    print('step2 done')

# get the accuracy
score = test(ctx_dl, spl_dl, labelled_data)
print('accuracy:', score)

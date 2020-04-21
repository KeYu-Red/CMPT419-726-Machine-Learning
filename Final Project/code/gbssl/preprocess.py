def prep(word):
    word = word.strip()
    word = word.replace('[', '')
    word = word.replace(']', '')
    return word


# read data from file
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


def preprocess_unlabel(data):
    data = [prep(word) for word in data]

    temp = []
    for word in data:
        if word != '' and '.pos' not in word:
            temp.append(word)

    sent_set = []
    sent = []
    for line in data:
        if line[0:2] == '==':
            if not sent:
                pass
            else:
                sent_set.append(sent)
                sent = []
        else:
            for word in line.split():
                sent.append(tuple(word.split('/')))

    return sent_set


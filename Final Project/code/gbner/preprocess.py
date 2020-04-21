def prep(word):
    word = word.strip()
    word = word.replace('[', '')
    word = word.replace(']', '')
    return word


# read data from file
def preprocess_label(data):
    sent_set = []
    sent = []
    for line in data:
        if line.split():
            sent.append(tuple(line.split()))
        else:
            if len(sent) >= 3:
                sent_set.append(sent)
            sent = []

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
                if len(sent) >= 3:
                    sent_set.append(sent)
                sent = []
        else:
            for word in line.split():
                sent.append(tuple(word.split('/')) + ('',''))

    return sent_set


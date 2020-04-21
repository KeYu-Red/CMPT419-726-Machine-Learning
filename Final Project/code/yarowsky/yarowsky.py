import random

def prep(word):
    word = word.replace('[', '')
    word = word.replace(']', '')
    word = word.strip()
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

def data_process(label_data):
    train_data=[]
    for test in label_data:
        spell=[]
        for i in range(len(test)):
            if test[i][3]!='O'and test[i][3]!='I-MISC':
                out=0
                for index in range(i+1,len(test)):
                    addWord1=''
                    if test[index][1]=='NNP':
                        addWord1=test[index][0]
                        break
                for index in range(i+1,len(test)):
                    addWord2=''
                    if test[index][1]=='VBZ':
                        addWord2=test[index][0]
                        break
                
                Entity = test[i][0] 
                if addWord1!='' and addWord2!='':
                    feature = [test[i][0], test[i][1],addWord1, addWord2]
                elif addWord1!=''and addWord2=='':
                    feature = [test[i][0], test[i][1],addWord1]
                elif addWord1==''and addWord2!='':
                    feature = [test[i][0], test[i][1],addWord2]
                else:
                    feature = [test[i][0], test[i][1]]
                Y_label = test[i][3]
                line = [Entity, feature,Y_label]
                train_data.append(line)   
    res = [] 
    for i in train_data: 
        if i not in res: 
            res.append(i) 
    return res

# def data_process(train_data):
#     train_data=[]
#     for test in label_data:
#         spell=[]
#         for i in range(len(test)):
#             if test[i][3]!='O'and test[i][3]!='I-MISC':
#                 
#                 Entity = test[i][0]
#                 feature = [test[i][0], test[i][1]]
#                 Y_label = test[i][3]
#                 line = [Entity, feature,Y_label]
#                 train_data.append(line)   
#     res = [] 
#     for i in train_data: 
#         if i not in res: 
#             res.append(i) 
#     return res

def H(count_xy, count_y):
    alpha=0.1
    k=3
    ans = 1.0*(count_xy+alpha)/(count_y+k*alpha)
    return ans
def CheckWhetherScore(train, test):
    score=0
    trainLen=len(train)
    testLen = len(test)
    if train[0]==test[0]:
        score=3
    for i in range(trainLen):
        for j in range(testLen):
            #print(train[i],test[j])
            if train[i]==test[j]:
                score = score+1
    limit = 1
    if score > limit:
        return True
    else: 
        return False
    
    
    
    
def yar_classifier(test_data,train_data):
    total=0
    score=0
    AddToTrain=[]
    Class_Label = ['I-LOC','I-ORG','I-PER']
    for index in range(len(test_data)):
        Count=[]
#         print(test_data[index])
        for labels in Class_Label:
            count_xy=0
            count_y=0
            for x in train_data:
                if x[2]==labels:
                    count_y=count_y + 1
                    if CheckWhetherScore(x[1],test_data[index][1]):
                        count_xy=count_xy + 1
            ans=H(count_xy, count_y)            
            One_test = [labels,[count_xy,count_y],[ans]]
            Count.append(One_test) 
        Count.append(test_data[index])
        choose=0
        if Count[0][2]<Count[1][2]:  
            choose=1
            if Count[1][2]<Count[2][2]:
                choose=2    
        if Count[0][2]<Count[2][2]:
            choose=2
        if Count[choose][0]==test_data[index][2]:
            score=score+1
            if random.randint(0,100)<75:
                AddToTrain.append(Count[3])
        total=total+1
    return [score/total, AddToTrain]

# def yar_classifier_2(test_data,train_data):
#     total=0
#     score=0
#     Class_Label = [['I-LOC',0],['I-ORG',1],['I-PER',2]]
#     for index in range(len(test_data)):
#         Count=[]
#         print(test_data[index])
#         for labels in Class_Label:
#             count_xy=0
#             count_y=0
#             for x in train_data:
#                 if x[2]==labels[0]:
#                     count_y=count_y + 1
#                     if (set(x[1]).issubset(set(train_data[labels[1]][1]))):
#                         count_xy=count_xy + 1
#             ans=H(count_xy, count_y)            
#             One_test = [labels,[count_xy,count_y],[ans]]
#             Count.append(One_test)  
#         choose=0
#         if Count[0][2]<Count[1][2]:  
#             choose=1
#             if Count[1][2]<Count[2][2]:
#                 choose=2    
#         if Count[0][2]<Count[2][2]:
#             choose=2
#         if Count[choose][0]==test_data[index][2]:
#             score=score+1
#         total=total+1
#     return [score,total,score/total]


def data_process_all(test_data):
    re=[]
    for word in test_data:
        for i in range (len(word)):
            spell=[word[i][0],[word[i][0],word[i][1],word[i][2]],word[i][3]]
            re.append(spell)
    return re

def FeatureCollct(Label_data):
    FeaturesCollection=[]
    ORG=[]
    PER=[]
    LOC=[]
    for word in Label_data:
        if word[2]=='I-LOC':
            LOC.append(word[1][0])
            LOC.append(word[1][1])
            LOC.append(word[1][2])
        elif word[2]=='I-ORG':
            ORG.append(word[1][0])
            ORG.append(word[1][1])
            ORG.append(word[1][2])
        elif word[2]=='I-PER':
            PER.append(word[1][0])
            PER.append(word[1][1])
            PER.append(word[1][2])
    FeaturesCollection=[['LOC',LOC],['ORG',ORG],['PER',PER]]
    return FeaturesCollection

def AddTrain(original, new):
    for word in new:
        original.append(word)
    return original

def randomizeSeeds(Label_data):
    len(Label_data)
    Train_Data=[]
    Test_Data=[]
    NUMBER=50
    for line in range(NUMBER):
        length=len(Label_data)
        num=random.randint(0,length)
        Train_Data.append(Label_data[num])
    return [Train_Data,Test_Data]

def testAccuracy(Train_Data, Label_data):
    i=0
    accuracy=0
    while True:
        i=i+1
        length=len(Train_Data)
        [accuracy,add]=yar_classifier(Label_data,Train_Data)
        new=AddTrain(Train_Data, add)
        Train_Data=[]
        for index in new: 
            if index not in Train_Data: 
                Train_Data.append(index)
        print("accuracy=",accuracy,"i=",i, "length of data_train", length)
        if(accuracy>0.90):
            break

def main():
    file="labeled_train"
    with open(file) as f:
        raw_data = f.readlines()
    label_data=preprocess_label(raw_data)
    Label_data = data_process(label_data)
    T=CheckWhetherScore(['Fischler', 'JJR', 'EU-wide', 'measures'],['Fischler', 'NNP'])
    print(T)
    [Train_Data,Test_Data]=randomizeSeeds(Label_data)
    testAccuracy(Train_Data, Label_data)

main()
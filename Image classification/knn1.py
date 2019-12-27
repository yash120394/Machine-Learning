import pandas as pd
import numpy as np
import random
import time
from scipy.spatial.distance import cdist
import operator
import sys

data = pd.read_table("/u/yashkuma/a4/"+sys.argv[1], delim_whitespace=True, header=None)
test = pd.read_table("/u/yashkuma/a4/"+sys.argv[2], delim_whitespace=True, header=None)

data.to_csv("/u/yashkuma/a4/knn.txt", header=None, index=None, sep=' ', mode='a')

train = pd.read_table("/u/yashkuma/a4/knn.txt", delim_whitespace=True, header=None)

def process_data(data):
    data1 = data.iloc[:,1:]
    key = data.iloc[:,0]
    return key, data1


def train_cv_split(data) :
    key, data = process_data(data)
    train = data.sample(frac=0.75,random_state=200)
    cv = data.drop(train.index)
    return train, cv


def choose_K(data):
    train, cv = train_cv_split(data)
    X_train = np.array(train.iloc[:,1:])
    Y_train = np.array(train.iloc[:,0])
    X_cv = np.array(cv.iloc[:,1:])
    Y_cv = np.array(cv.iloc[:,0])
    dist_matrix = cdist(X_cv, X_train)
    accuracy = 0 
    K = [91,101,111]
    for k in K:
        Y_cv_pred = []
        for i in range(len(dist_matrix)):
            closest = np.argsort(dist_matrix[i])[:k]
            class1 = []
            for j in range(len(closest)):
                class1.append(Y_train[closest[j]])
            d = {x:class1.count(x) for x in class1}
            c = max(d.items(), key=operator.itemgetter(1))[0]
            Y_cv_pred.append(c)
        Y_cv1 = Y_cv.tolist()
        accuracy1 = sum(1 for x,y in zip(Y_cv1,Y_cv_pred) if x == y) / len(Y_cv)
        if accuracy1 > accuracy:
            accuracy = accuracy1
            k_opt = k 
        else :
            continue
    return k_opt

     


def knn(train, test):
    train_key,train_data = process_data(train)
    test_key,test_data = process_data(test)
    test_label = np.array(test_data.iloc[:,0])
    test_predictor = np.array(test_data.iloc[:,1:])
    train_label = np.array(train_data.iloc[:,0])
    train_predictor = np.array(train_data.iloc[:,1:])
    dist_matrix =  cdist(test_predictor, train_predictor)
    test_label_pred = []
    k_opt = choose_K(train)
    for i in range(len(dist_matrix)):
        k_small = np.argsort(dist_matrix[i])[:k_opt]
        class1 = []
        for j in range(len(k_small)):
            class1.append(train_label[k_small[j]])
        d = {x:class1.count(x) for x in class1}
        c = max(d.items(), key=operator.itemgetter(1))[0]
        test_label_pred.append(c)
    test_label1 =  test_label.tolist()
    accuracy_test = sum(1 for x,y in zip(test_label1,test_label_pred) if x == y) / len(test_label1)
    test_key1 = test_key.tolist()
    pred_df = pd.DataFrame({'image_id':test_key1,'label':test_label1,'pred_label':test_label_pred})
    return pred_df,accuracy_test


pred, accuracy = knn(train,test)
print("Classification accuracy")
print(accuracy*100)


file = open("/u/yashkuma/a4/"+sys.argv[3], 'a')
file.write(pred.to_string())
file.close()
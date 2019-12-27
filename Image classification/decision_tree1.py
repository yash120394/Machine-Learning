import pandas as pd
import numpy as np
import time
import random
from scipy import stats
import sys

data = pd.read_table("/u/yashkuma/a4/"+sys.argv[1], delim_whitespace=True, header=None)
test = pd.read_table("/u/yashkuma/a4/"+sys.argv[2], delim_whitespace=True, header=None)

def process_data(data):
    data1 = data.iloc[:,1:]
    key = data.iloc[:,0]
    return np.array(key), np.array(data1)

key, data1 = process_data(data)


def calculate_gini(y):
    unique, counts = np.unique(y, return_counts=True)
    sum_sqr = np.sum([(counts[i]/sum(counts))**2 for i in range(len(unique))])
    gini = 1 - sum_sqr
    return gini
    

def calculate_info_gain(x,y):
    total_gini = calculate_gini(y)
    unique_y, counts_y = np.unique(y, return_counts=True)
    random_x = random.sample(range(min(x), max(x)), 5)
    data = np.column_stack((x,y))
    best_gain = 0
    a = x
    b = y
    for k in random_x:
        q_l = len(a[a <= k])/len(a)
        q_r = len(a[a > k])/len(a)
        subset_l = data[data[:,0] <= k ,:]
        subset_r = data[data[:,0] > k ,:]
        x_l = subset_l[:,0]
        y_l = subset_l[:,1]
        x_r = subset_r[:,0]
        y_r = subset_r[:,1]
        unique_l, counts_l = np.unique(y_l, return_counts=True)
        gini_l = 1 - np.sum([(counts_l[i]/sum(counts_l))**2 for i in range(len(unique_l))])
        unique_r, counts_r = np.unique(y_r, return_counts=True)
        gini_r = 1 - np.sum([(counts_r[j]/sum(counts_r))**2 for j in range(len(unique_r))])
        info_gain = total_gini - (q_l*gini_l + q_r*gini_r)
        if  info_gain > best_gain:
            best_gain = info_gain
            split_value = k
        else:
            continue
    return best_gain, split_value


  
def Decision_tree_classifier(y_sub,x_sub,y,x,d):
    d = d+1
    if len(y_sub) == 0:
        return np.unique(y)[np.argmax(np.unique(y,return_counts=True)[1])]
    
    elif d == 5:
        return np.unique(y_sub)[np.argmax(np.unique(y_sub,return_counts=True)[1])]    
    
    else:
        best_info_gain = 0
        for i in range(len(x_sub[0])):
            gain, split_col_value = calculate_info_gain(x_sub[:,i],y_sub)
            if gain > best_info_gain:
                best_info_gain = gain
                split_value1 = split_col_value
                col_ind = i
            else :
                continue
        best_col = col_ind
        tree = {best_col:{}}
        data = np.column_stack((y_sub,x_sub))
        X = x_sub
        Y = y_sub
        a= [data[data[:,best_col] <= split_value1,:],data[data[:,best_col] > split_value1,:]]
        for j in range(len(a)):
            part = a[j]
            sub_x = part[:,1:]
            sub_y = part[:,0]
            sub_tree = Decision_tree_classifier(sub_y,sub_x,Y,X,d)
            tree[best_col][str(split_value1)+'_'+str(j)] = sub_tree
        return (tree)
 

tree = Decision_tree_classifier(data1[:,0],data1[:,1:],data1[:,0],data1[:,1:],0)
f = open("/u/yashkuma/a4/tree.txt","w") 
print(tree,file=f)



def predict(query,tree,default=0):
    for key in list(query.keys()):
        if key in tree.keys():
            try:
                z = list(tree[key])[0]
                n = int(z.rsplit('_', 1)[0])
                if query[key] > n:
                    result = tree[key][str(n)+'_'+str(1)]
                else:
                    result = tree[key][str(n)+'_'+str(0)]
            except:
                return 0
        
            result1 = result
            if isinstance(result1,dict):
                return predict(query,result)
            else:
                return result1
            
            
            
def test_predict(data,tree):
    data1 = data.iloc[:,2:]
    new_col = [i for i in range(len(data1.columns))]
    data1.columns = new_col
    queries = data1.to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"])
    for i in range(len(data1)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data.iloc[:,1])/len(data))*100,'%') 
    return predicted
    

f = open("/u/yashkuma/a4/tree.txt", "r")
tree1 = eval(f.read())
key1 = test.iloc[:,0]
actual = test.iloc[:,1]
pred = test_predict(test,tree1)

pred_df = pd.concat([key1,actual,pred],axis=1)


file = open("/u/yashkuma/a4/"+sys.argv[3], 'a')
file.write(pred_df.to_string())
file.close()

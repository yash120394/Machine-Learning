For running decision_tree1.py

  
```bash
python3 decision_tree1.py train-data.txt test-data.txt output-dt.txt
```

For running knn1.py
 
```bash
python3 knn1.py train-data.txt test-data.txt output-knn.txt
```


# Part1
## KNN

### Optimisation
- Splitting train data into train and cross validation set 
- Initialised a list of K which is approx is sqrt(n) where n is number of samples in the dataset
- For each K, performed KNN to find best accuracy on cv and return optimal K
- For simplication and less computational cost, used only one cross validation set which is 25% of the train data
- Used scipy library to calculate distance matrix between train and cv set and iterated over rows of the distance matrix where each row is the row index of cv, and column index is row index of train data
- For each row of cv, calculated K closest distance from train rows and appended the class of that into a list
- Calculated mode of class in that list and assigned that class for the cv row

Here is the sample code for choosing K 

```python
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
 ```

### KNN Algorithm
- Used optimised K to predict class on test data
- In the knn function, all the steps mentioned above is same except it uses all train data as model input and predict class on test data 
- Returned prediction dataframe and accuracy of the model from knn function
- Prediction dataframe consist of image url, original class and predicted class

### Simplification
- Initialised K list of only 3 element for choosing optimum K because of high computational cost
- For getting mode of K nearest classes, if two classes are mode, then choosing class which appears first as prediction class for that test row

### Challenges
- High computational cost for calculating distance of each row of test data with all train data rows
- Choosing optimum K is a NP hard problem, only reasonable solution is to use cross validation set to get optimum K 


# Part 2
## Decision Tree Classifier

### Gini and Info gain calculation
- Given response variable that is class used gini index which is 1 - sum(p^) where p is probability of each class
- For calculating optimal split of a column and best info gain, randomly chose 5 values from a column within the range of that column, to calculate info gain and chose highest info gain as the info gain for that column and value which gave the highest info gain as the split value of that column

Here is the sample functions for gini and info gain calculation :

```python
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
```

### Decision Tree 
- Used recursion method to create tree nodes
- Expanded the depth till 6 because of high computational cost
- Stopping criteria :
1. If the length of data at any node is empty, then returned max value of unique classes of its parent node
2. If depth reaches 6, then returned max value of unique class of the current node
- At each node, calculated which column is giving the best info gain and appended that column index as a dictionary key
- For column which is giving the best split, data has been split into two halves based on the split value of that column and appended both split data into a list
- Iterated over a list and performed recursion method to expand nodes for each splits till stopping criteria is met
- Appended the tree of each halves of the data as a value for the column over which split has taken place

Sample code for Decision tree classifier

```python
def Decision_tree_classifier(y_sub,x_sub,y,x,d):
    d = d+1
    if len(y_sub) == 0:
        return np.unique(y)[np.argmax(np.unique(y,return_counts=True)[1])]
    
    elif d == 6:
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
 ```
 

### Prediction
- Created dictionary query of test data with key as column index of features
- For each key i.e. column index in a test data dictionary, if the key existed in the model tree dictionary, used the key of tree to expand to next item in the dictionary to get split value and based on split value and value of key in test data dictionary, returned result the value of item if there is no sub items else used the result to check whether it is an instance. If so, then used recursion to expand the next key

### Simplification 
- Used 5 random values of a column to get the best split value of that column because of high computational cost
- Hard coded depth = 5 in the decision tree function for stopping criteria

### Challenges 
- High computational cost in finding the best split value of each column at each iteration
- High computational cost for searching tree at depth more than 5


# Overall Classifier comparision
-----------------------------------------------
- Accuracy for KNN on test data : 72%
- Accuracy for Decision tree on test data : 60%
-----------------------------------------------
- Running time for KNN : 15 mins
    - Running time increases linearly as we increase K 
    - If we increase number of K's to choose optimum K, running time increases exponentially
    - Running time will increase exponentially in case of KNN if we increase data size as it has to calculate distance of each point in test data with entire train data

-----------------------------------------------
- Running time for Decision Tree : 5 mins
    - Training time is 3.5 mins while testing is done within 1 min
    - As we increase number of depths, running time increases exponentially
    - Running time increases if we iterate over each unique column values for getting best split value of a column. Depends on how many unique values are there in a column
    - Running time will increase but not that much as compared to KNN if we increase data size. However it will increase exponentially if we increase the number of features used in building a classifier
    - Pruning the tree will reduce the running time of model which can be done as a next step in this assignment

-----------------------------------------
Sample correct clasified images for KNN
1. 10008707066.jpg  : 0 
2. 10107730656.jpg  : 0
3. 10353444674.jpg  : 270 

Sample incorrect classified images for KNN
1. 10351347465.jpg : Class - 270, Pred Class - 180
2. 11057679623.jpg : Class - 0, Pred Class - 180
3. 1160014608.jpg  : Class - 180, Pred Class - 90

----------------------------------------
Sample correct clasified images for Decision Tree
1. 10008707066.jpg  : 0 
2. 10107730656.jpg  : 180
3. 10161556064.jpg  : 270 

Sample incorrect classified images for Decision Tree
1. 10352491496.jpg : Class - 90, Pred Class - 270
2. 10484444553.jpg : Class - 180, Pred Class - 0
3. 10931472764.jpg  : Class - 0, Pred Class - 270

-----------------------------------------
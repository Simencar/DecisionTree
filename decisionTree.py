"""
This is a module for a decision tree with continuous features

Typical usage:

from decisionTree import Tree

dt = Tree()
dt.learn(train_features, train_labels)
dt.predict(test_feature)
dt.score(test_features, test_labels)
"""

from math import inf, log, ceil
import numpy as np 
from statistics import mean



class Node:
    def __init__(self, mean, col, labels, left, right):
        self.mean = mean
        self.col = col
        self.labels = labels
        self.left = left
        self.right = right


class Leaf:
    def __init__(self, label):
        self.label = label


class Tree:
    def __init__(self):
        self.root = None

   
    def learn(self, X, y, impurity_measure="entropy", prune=False, prune_size=0.2): 
        """
        Build the decision tree. Updates the root variable to point at the root of the built tree
        """
        if prune_size < 0 or prune_size > 1:
            raise ValueError("prune_size must be between 0 and 1") 
        if not impurity_measure in ["entropy", "gini"]:
            raise ValueError(impurity_measure +" is not a valid impurity_measure") 
        #create pruning data
        if prune: 
            #pruning data will be taken from 0 to "size" in the training data
            #training data will be updated with all data after "size"
            size = ceil(len(y)*prune_size)
            px = X[:size,:] 
            py = y[:size]   
            X = X[size:,:]  
            y = y[size:]    
        tree = build_tree(X, y, impurity_measure) #build initial tree
        if prune:
            tree = prune_tree(tree, px, py) #prune the tree
        self.root = tree 



    def predict(self, x):
        """
        predict label for a single datapoint
        """
        return predict_label(x, self.root)



    def score(self, X, y):
        """
        Calculate the accuracy of prediction on data X by comparing to correct labels in y 
        """
        return accuracy(self.root, X, y)

    

#------------------------Helper functions--------------------------------------------

def entropy(labels):
    """
    calculate entropy for a set of labels
    """
    value,counts = np.unique(labels, return_counts=True) #find labels and their respective count
    if len(value)==1: #only 1 label
        return 0
    probs = counts/len(labels) #find probability for each label
    ent = 0
    for p in probs: #calculate the entropy
        ent += p*log(p,2)
    return -ent

#entropy and gini functions can be merged for a smaller code-base. I kept them split for readability. 

def gini(labels):
    """
    calculate gini index for a set of labels
    """
    value,counts = np.unique(labels, return_counts=True) #find labels and their respective count
    if len(value)==1: #only 1 label
        return 0
    probs = counts/len(labels)
    gini = 0
    for p in probs:
        gini += p*p
    return 1-gini




#NOTE: I chose to minimize the conditional entropy to find the best split(when entropy is selected). This avoids an entropy function-call on the whole set of labels,
#and is equivalent to finding and maximizing information gain.
def find_best_split(X, y, impurity_measure):
    """
    finds best split by calculating conditional entropy or gini index for all splits, based on given impurity measure
    returns: mean of the feature-column that gives lowest impurity when split, column number of that feature in X
    """
    lowest_impurity = inf
    best_mean = 0 #mean of feature split with lowest impurity
    best_col = 0 #column number of the feature that gives lowest impurity when split
    rows, cols = X.shape
    for c in range(cols):
        y_less = [] #labels for datapoints < mean
        y_more = [] #labels for datapoints >= mean
        col_mean = mean(X[:,c]) #mean of column c
        for r in range(rows):
            #store label based on current feature value compared against mean
            if(X[r,c] < col_mean): 
                y_less.append(y[r])
            else:                         
                y_more.append(y[r])  
        imp = 0
        #probability that feature value is less/more than mean
        prob_less = len(y_less)/len(y) 
        prob_more = len(y_more)/len(y)

        #calculate impurity with given impurity measure. low impurity = good split
        if impurity_measure == 'entropy':
            entropy_less = entropy(y_less)
            entropy_more = entropy(y_more)
            imp = prob_less*entropy_less + prob_more*entropy_more

        else:
            gini_less = gini(y_less)
            gini_more = gini(y_more)
            imp = prob_less*gini_less + prob_more*gini_more

        #update if this split has lower impurity
        if imp < lowest_impurity:
            lowest_impurity = imp
            best_mean = col_mean
            best_col = c
    return best_mean, best_col


 
def split_data(X, y, mean, col):
    """
    splits the data in a left and right part(used in nodes), based on the column number of the feature we want to split, and the mean of that column
    returns: X_left, X_right, Y_left, Y_right
    """
    X_left = []
    X_right = []
    Y_left = []
    Y_right = []
    #append to the appropriate list based on the value in the current row and column compared to the mean
    for r in range(len(y)):
        if X[r,col] < mean:
            X_left.append(X[r,:])
            Y_left.append(y[r])
        else:
            X_right.append(X[r,:])
            Y_right.append(y[r])
    return np.array(X_left), np.array(X_right), np.array(Y_left), np.array(Y_right) 



def has_one_label(y):
    """
    Checks if a list of labels only contain one label type
    returns: True if 1 label
    """
    value = np.unique(y)
    return len(value) == 1



def has_identical_features(X):
    """
    Checks if the feature values are identical for every column
    returns: True if identical values
    """
    rows, cols = X.shape
    for c in range(cols):
        x = X[0,c]
        for r in range(rows):
            if x != X[r, c]:
                return False
    return True



def majorityLabel(y):
    """
    Finds the most common label in a list
    """
    values, counts = np.unique(y, return_counts=True)
    index = np.argmax(counts)
    return values[index]



def build_tree(X, y, impurity_measure):
    """
    Recursively builds the decision tree with given training data and impurity measure
    returns: root of the tree. Either Node or a leaf(if no splits were made)
    """
    if has_one_label(y): #every data point has the same label, return a leaf with that label
        return Leaf(y[0])

    elif has_identical_features(X): #if all data points has identical feature values, return leaf with with the most common label 
        label = majorityLabel(y)
        return Leaf(label)

    else: #need to split
        mean, col = find_best_split(X, y, impurity_measure) #find the best feature to split and return the mean and column number of that feature
        X_left, X_right, Y_left, Y_right = split_data(X, y, mean, col) #split data based on calculated mean and column
        #build left and right subtree recursievly
        left = build_tree(X_left, Y_left, impurity_measure)
        right = build_tree(X_right, Y_right, impurity_measure)
        return Node(mean, col, y, left, right)
        


def predict_label(x, tree):
    """
    Predicts the labels for a single data point
    returns: the label
    """
    if tree == None:
        raise TypeError("Decision tree not initialized. Build with learn() before predicting")
    if type(tree).__name__ == "Leaf": #reached leaf, return label
        return tree.label
    else:
        if x[tree.col] < tree.mean: 
            return predict_label(x, tree.left) #less than mean, go left
        else: 
            return predict_label(x, tree.right) #more than or equal mean, go right



def accuracy(tree, X, y):
    """
    calculates the accuracy of predicting labels by comparing to the correct labels
    """
    n = len(y)
    n_correct = 0
    for x in range(n):
        label = predict_label(X[x,:], tree)
        if label == y[x]: #labels match
            n_correct+=1
    return n_correct/n 



def prune_tree(tree, px, py):
    """
    Prunes the decision tree in a bottom up fashion, from the leafs
    returns: the pruned tree
    """
    if len(py) == 0: #pruning data is empty and we cant calculate accuracy.
        return tree
    elif type(tree).__name__ == "Leaf": #reached leaf, go back
        return tree
    else:
        #split the pruning data for left and right child based on the feature that is split and the mean of that feature column
        px_left, px_right, py_left, py_right = split_data(px, py, tree.mean, tree.col) 
        #Prune left and right child
        tree.left = prune_tree(tree.left, px_left, py_left)
        tree.right = prune_tree(tree.right, px_right, py_right)

        #children of this node has been pruned. Now we can prune this node
        m_label = majorityLabel(tree.labels) #find majority label from original training labels in this node
        full_accuracy = accuracy(tree, px, py) #calculate the accuracy of the pruning data on the full tree
        prune_accuracy = accuracy(Leaf(m_label), px, py) #calculate the accuracy of the pruning data if this node is set to a leaf with majority label

        if prune_accuracy >= full_accuracy: 
            return Leaf(m_label) #replace this node with a leaf with majority label if accuracy does not decrease on pruning data 
        return tree






   



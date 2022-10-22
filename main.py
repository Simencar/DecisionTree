import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from decisionTree import Tree
import time



#------------------ Load and split data ------------------

data = pd.read_csv("C:\\Users\\homse\Desktop\\DecisionTree\\magic04.data", header=None).to_numpy()
X = data[:,:10]
y = data[:,10]


seed = 666                    # Fix random seed for reproducibility
# Shuffle and split the data into train and a concatenation of validation and test sets
X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, y,      
                                       test_size=0.4, 
                                       shuffle=True, 
                                       random_state=seed)

seed = 221
# Shuffle and split the data into validation and test sets
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test,   
                                       test_size=0.5, 
                                       shuffle=True, 
                                       random_state=seed) 



#-------------Model Training and prediction----------------

decisionTree = Tree()
clf = DecisionTreeClassifier(criterion="entropy", random_state=0) #sklearn

#-------------------Entropy-------------------
decisionTree.learn(X_train, Y_train)
train_acc_ent = decisionTree.score(X_train, Y_train)
val_acc_ent = decisionTree.score(X_val, Y_val)
#sklearn
clf.fit(X_train,Y_train)
sklearn_entropy_val_acc = clf.score(X_val, Y_val) 

#------------------Gini_index-----------------
decisionTree.learn(X_train, Y_train, "gini")
train_acc_gini = decisionTree.score(X_train, Y_train)
val_acc_gini = decisionTree.score(X_val, Y_val)
#sklearn
clf.criterion="gini"
clf.fit(X_train,Y_train)
sklearn_gini_val_acc = clf.score(X_val, Y_val) 

#---------------Entropy and prune-------------
decisionTree.learn(X_train, Y_train, prune=True)
train_acc_ent_prune = decisionTree.score(X_train, Y_train)
val_acc_ent_prune = decisionTree.score(X_val, Y_val)
test_acc_ent_prune = decisionTree.score(X_test, Y_test) #<- Selected model for test set

#--------------Gini_index and prune-----------
decisionTree.learn(X_train, Y_train, "gini", prune=True)
train_acc_gini_prune = decisionTree.score(X_train, Y_train)
val_acc_gini_prune = decisionTree.score(X_val, Y_val)



print("------------ TRAINING ACCURACY ----------")
print("Entropy             :" , train_acc_ent)
print("Gini_index          :" , train_acc_gini)
print("Entropy and prune   :" , train_acc_ent_prune)
print("Gini_index and prune:" , train_acc_gini_prune)
print("")

print("------------ VALIDATION ACCURACY --------")
print("Entropy             :" , val_acc_ent)
print("Sklearn entropy     :" , sklearn_entropy_val_acc)
print("Gini_index          :" , val_acc_gini)
print("Sklearn Gini_index  :" , sklearn_gini_val_acc)
print("Entropy and prune   :" , val_acc_ent_prune)
print("Gini_index and prune:" , val_acc_gini_prune)
print("")

print("------------ TEST ACCURACY --------------")
print("Entropy and prune   :",  test_acc_ent_prune)
print("")

print("------------- EXECUTION TIME ------------")
start = time.time()
clf.fit(X_train, Y_train)
end = time.time()
print("Sklearn build tree  :", end-start)
start = time.time()
decisionTree.learn(X_train, Y_train, prune=True)
end = time.time()
print("My build tree       :", end-start)
start = time.time()
clf.predict(X_val)
end = time.time()
print("Sklearn predict     :", end-start)
start = time.time()
for r in range(len(Y_val)):
    decisionTree.predict(X_val[r,:])
end = time.time()
print("My predict          :", end-start)


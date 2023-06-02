#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


col_names = ['Type', 'Alcohol', 'MalicAcid',
             'Ash', 'Alcalinity', 'Magnesium','Phenols','Flavanoids','Nonflavanoid',
             'Proanthocyanins','Colorlntensity','Hue','DilutedWines','Proline']
data = pd.read_csv('wines.csv',skiprows=1, header=None ,names=col_names, skip_blank_lines=True, na_values='?', sep=',')
data.head(40)


# In[3]:


len(data)


# In[4]:


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        #constructor for the node
        #tree node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        #leave node
        self.value = value


# In[6]:


class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        #constructor for the DecisionTreeClassifier class

        #stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        #initialise the root of the tree
        self.root = None

    def build_tree(self, dataset, curr_depth=0):
        #builds the decision tree recursively
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        #split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            #find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            #check if information gain is positive
            if best_split["info_gain"]>0:
                #recurse on left and right subtrees
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["info_gain"])
        #compute leaf node value
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        #finds the best split for a dataset
        best_split = {}
        max_info_gain = -float("inf")
        #loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            #loop over all the feature values present in the data
            for threshold in possible_thresholds:
                #get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                #check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    #compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    #update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        #return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        #splits the dataset based on feature and threshold
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        #computes information gain
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        #computes entropy of label array
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y==cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        #computes gini index of label array
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y==cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    
    def calculate_leaf_value(self, Y):
        #computes the leaf node
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        #prints the tree in a readable format
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        #builds the tree
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        #predicts the class labels
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        #predicts a single data point
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    


# In[97]:


from sklearn.model_selection import train_test_split
# Train-Test Split
X = data.iloc[:,1:].values
Y = data.iloc[:,0].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=42)


# In[98]:


X


# In[99]:


from sklearn.model_selection import cross_val_score
classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=2)
classifier.fit(X_train, Y_train)
classifier.print_tree()


# In[100]:


from sklearn.metrics import accuracy_score
Y_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)


# In[96]:


from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

# Assuming X and Y are the data
kf = KFold(n_splits=5, random_state=17, shuffle=True)

metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=3)
    classifier.fit(X_train, Y_train)
    
    Y_pred = classifier.predict(X_test)

    # classifier.print_tree()
    # acc = accuracy_score(Y_test, Y_pred)
    # print("Accuracy:", acc)
    
    metrics["accuracy"].append(accuracy_score(Y_test, Y_pred))
    metrics["precision"].append(precision_score(Y_test, Y_pred, average='macro', zero_division=0))
    metrics["recall"].append(recall_score(Y_test, Y_pred, average='macro', zero_division=0))
    metrics["f1"].append(f1_score(Y_test, Y_pred, average='macro', zero_division=0))

# The scores for each fold
for metric, scores in metrics.items():
    print(f"{metric.capitalize()} scores: ", scores)
    print(f"Average {metric}: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores) * 2))


# In[ ]:





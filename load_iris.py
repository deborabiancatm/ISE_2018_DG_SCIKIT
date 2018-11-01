# -*- coding: utf-8 -*-
import numpy as np
import graphviz
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

#imprime o dataset
'''
#import dataset
for i in range(len(iris.target)):
    print ("Example",i,": label",iris.target[i],", features", iris.data[i])
'''

#train a classifier
test_idx = [1, 51, 101]

#training data
training_target = np.delete(iris.target, test_idx)
training_data = np.delete(iris.data, test_idx, axis = 0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(training_data, training_target)

#predict label to a new flower
print ("Data set ID:", test_idx)
print ("Data set classification:", test_target)
print ("Data test classification:", clf.predict(test_data))
print ("[0]Iris setosa\t[1]Iris versicolor\t[2]Iris virginica")

#Gera o arquivo em pdf da árvore de decisões para classificar as flores
dot_data = tree.export_graphviz (clf, out_file = None,
                     feature_names = iris.feature_names,
                     class_names = iris.target_names,
                     filled = True, rounded = True,
                     special_characters = True)
graph = graphviz.Source(dot_data)  
graph.render("iris") 
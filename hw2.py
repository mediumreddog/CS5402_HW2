from id3 import Id3Estimator, export_graphviz
import numpy as np
import pandas as pd
from c45 import C45
from sklearn import tree
import graphviz

def id3():
  headers = pd.read_csv('Task4_Data.csv', nrows=1).columns.values
  headers = headers[3:6]

  X = pd.read_csv('Task4_Data.csv').values
  y = X[:,6]
  X = X[:,3:6]

  clf = Id3Estimator()
  clf.fit(X, y, check_input=True)
  export_graphviz(clf.tree_, 'tree.dot', headers)

def c4dot5():
  c1 = C45("./Task4_Data.csv", "./Task4_Data.csv",)
  c1.fetchData()
  c1.preprocessData()
  c1.generateTree()
  c1.printTree()

def cart():
  headers = pd.read_csv('Task4_Data.csv', nrows=1).columns.values
  headers = headers[3:6]

  X = pd.read_csv('Task4_Data.csv').values
  y = X[:,6]
  X = X[:,3:6]

  clf = tree.DecisionTreeClassifier()
  clf = clf.fit(X, y)

  dot_data = tree.export_graphviz(clf, out_file=None) 
  graph = graphviz.Source(dot_data) 
  graph.render("task4-cart") 


#id3()
#c4dot5()
#cart()
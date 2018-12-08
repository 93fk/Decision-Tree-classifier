NAME = '0_first_notebook' ## Name of the notebook goes here (without the file extension!)
PROJECT = 'Decision Tree classifier'
PYTHON_VERSION = '3.6.7'

## imports
import os, re
import pandas as pd
import seaborn as sns
import graphviz
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

## Set working directory  
workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)


##--CODE--##


data = datasets.load_iris()
target_names = data.target_names
target = data.target
col_names = data.feature_names

iris = pd.DataFrame(data.data, columns=col_names)
species_dict = {0:'setosa',  1:'versicolor', 2:'virginica'}
species_list = [species_dict[x] for x in target]

iris['species'] = species_list

sns.pairplot(iris, hue='species')

iris_extract = iris[['petal length (cm)', 'petal width (cm)']]

DTC = DecisionTreeClassifier(max_depth=3).fit(iris_extract, target)
dot_data = export_graphviz(DTC, out_file=None, feature_names=['petal length (cm)', 'petal width (cm)'], class_names=['setosa', 'versicolor', 'virginica'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('dtree_render', view=True)
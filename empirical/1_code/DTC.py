NAME = '0_first_notebook' ## Name of the notebook goes here (without the file extension!)
PROJECT = 'Decision Tree classifier'
PYTHON_VERSION = '3.6.7'

## imports
import os, re
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


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

DTC = DecisionTreeClassifier(max_depth=2).fit(iris_extract, target)

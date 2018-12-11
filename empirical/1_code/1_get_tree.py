NAME = '1_get_tree' ## Name of the notebook goes here (without the file extension!)
PROJECT = 'Decision Tree Classifier'
PYTHON_VERSION = '3.6.7' 

## Imports
import os
import pandas as pd
import pydotplus
from sklearn.trees import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image

## Set working directory  
workdir = '/home/filip/Git/'+PROJECT
os.chdir(workdir)

## Set  up pipeline folder if missing  
if os.path.exists(os.path.join('empirical', '2_pipeline')):
    pipeline = os.path.join('empirical', '2_pipeline', NAME)
else:
    pipeline = os.path.join('2_pipeline', NAME)
    
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))
              
        
## Code
df = pd.read_csv(workdir+'/empirical/2_pipeline/0_feature_candidates/features.csv', usecols=[1,2,3])
df.columns = ['feature_1', 'feature_2', 'target']

DTC = DecisionTreeClassifier(max_depth=2).fit(df.iloc[:,0:2], df.iloc[:,2])
graph = export_graphviz(DTC)
viz = pydotplus.graph_from_dot_data(graph)
Image(viz.create_png())
## TODO - create df tith vlines and hlines
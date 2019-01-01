NAME = '1_get_tree' ## Name of the notebook goes here (without the file extension!)
PROJECT = 'Decision Tree Classifier'
PYTHON_VERSION = '3.6.7' 

## Imports
import os
import re
import pickle
import pandas as pd
import pydotplus
from sklearn.tree import DecisionTreeClassifier
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
dims = [(df['feature_1'].min(), df['feature_1'].max()), (df['feature_2'].min(), df['feature_2'].max())]

DTC = DecisionTreeClassifier(max_depth=2).fit(df.iloc[:,0:2], df.iloc[:,2])
graph = export_graphviz(DTC)
viz = pydotplus.graph_from_dot_data(graph)
Image(viz.create_png())

pattern_0 = re.compile(r'(X\[0\]) [=<>]* ([0-9.]*)')
pattern_1 = re.compile(r'(X\[1\]) [=<>]* ([0-9.]*)')

vlines = [p[1] for p in pattern_0.findall(graph)]
hlines = [p[1] for p in pattern_1.findall(graph)]

#save vlines and hlines on disk

pickle.dump(vlines, open(workdir+'/empirical/2_pipeline/'+NAME+'/vlines.obj', 'wb'))
pickle.dump(hlines, open(workdir+'/empirical/2_pipeline/'+NAME+'/hlines.obj', 'wb'))

pickle.dump(DTC, open(workdir+'/empirical/2_pipeline/'+NAME+'/DTC.obj', 'wb'))

pickle.dump(dims, open(workdir+'/empirical/2_pipeline/'+NAME+'/dims.obj', 'wb'))
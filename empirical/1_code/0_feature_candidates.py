NAME = '0_feature_candidates' ## Name of the notebook goes here (without the file extension!)
PROJECT = 'Decision Tree Classifier'
PYTHON_VERSION = '3.6.7'

## Imports
import os
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

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
data = datasets.load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

DTC = DecisionTreeClassifier().fit(df.drop('target', axis=1), df['target'])

best_predictors = pd.DataFrame(DTC.feature_importances_, index=df.drop('target', axis=1).columns)
feature_names = list(best_predictors.sort_values(by=0, ascending=False).index[:2])
feature_names.append('target')

output_df = df[feature_names]

output_df.to_csv(path_or_buf=workdir+'/empirical/2_pipeline/'+NAME+'/features.csv')
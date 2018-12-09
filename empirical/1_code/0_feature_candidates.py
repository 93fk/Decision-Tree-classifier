NAME = '0_feature_candidates' ## Name of the notebook goes here (without the file extension!)
PROJECT = 'Decision Tree Classifier'
PYTHON_VERSION = '3.6.7'

## imports
import os
import pandas as pd
import seaborn as sns
from sklearn import datasets

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


##--CODE--##
data = datasets.load_iris()
iris = pd.DataFrame(data.data, columns=data.feature_names)
iris['species'] = data.target

species_dict = {0:data.target_names[0], 1:data.target_names[1], 2:data.target_names[2]}
iris['species'] = iris['species'].map(species_dict)

iris_pairplot = sns.pairplot(iris, hue='species')
os.chdir(workdir+'/empirical/2_pipeline/'+NAME)
iris_pairplot.savefig('iris_pairplot.png')
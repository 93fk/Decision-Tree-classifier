NAME = '2_plot_decision_tree' ## Name of the notebook goes here (without the file extension!)
PROJECT = 'Decision Tree Classifier'
PYTHON_VERSION = '3.6.7'

## Imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
DTC = pickle.load(open(workdir+'/empirical/2_pipeline/1_get_tree/DTC.obj', 'rb'))

vlines = pickle.load(open(workdir+'/empirical/2_pipeline/1_get_tree/vlines.obj', 'rb'))
hlines = pickle.load(open(workdir+'/empirical/2_pipeline/1_get_tree/hlines.obj', 'rb'))

dims = pickle.load(open(workdir+'/empirical/2_pipeline/1_get_tree/dims.obj', 'rb'))

x_scale = np.linspace(dims[0][0], dims[0][1], 100)
y_scale = np.linspace(dims[1][0], dims[1][1], 100)

mesh = np.meshgrid(x_scale, y_scale)
mesh = list(zip(mesh[0].flatten(), mesh[1].flatten()))

predicted = DTC.predict(mesh)
predicted = predicted.reshape(100, 100)

#rescale horizontal and vertical lines to match imshow dimensions (100x100)
def rescale(value, scale):
    _ = []
    low = float(min(scale))
    high = float(max(scale))
    try:
        for v in value:
            _.append(100*(float(v)-low)/(high-low))
    except:
        return None
    return _

#plot figure
plt.imshow(predicted)
for v in rescale(vlines, dims[0]):
    plt.axvline(v, color='r', linestyle='--')
for h in rescale(hlines, dims[1]):
    plt.axhline(h, color='r', linestyle='--')
#remove ticks and labels
plt.tick_params(axis='both', bottom=False, left=False)
plt.gca().set_yticklabels([])
plt.gca().set_xticklabels([])

plt.savefig(workdir+'/empirical/3_output/results/plot.png')

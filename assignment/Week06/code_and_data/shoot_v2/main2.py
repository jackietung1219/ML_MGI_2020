#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVR
from sklearn.manifold import Isomap
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import time
from data_generator import *
#%%
# Generating data 
velocity_init = 0
velocity_end = 100
theta_init = 0
theta_end = 90
sample_amount = 10000
data_g(velocity_init, velocity_end, theta_init, theta_end, sample_amount)
#%%
# Setting training parameter
from _tmp import parameter
cv=5
scoring='r2'
train_sizes=[1e-3, 1e-2, 1e-1, 1e0]
parameter(cv, scoring, train_sizes)
#%%
# Choosing ML model 
from TrainModels2 import * # RFr(), SVr(), MLPr_3(), MLPr_i3(), MLPr_i7(), MLPr_i11()
MLPr_i11()
#%%
# 3D ploting (including outlier data)
from ploting2 import *
v_start = 0
v_end = 200
theta_start = 0
theta_end = 180
seed_amount = 250
data_ploting_parameter(v_start, v_end, theta_start, theta_end, seed_amount)
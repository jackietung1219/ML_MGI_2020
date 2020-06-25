# %%
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
from TrainModels import *  # RFr(), SVr(), MLPr_3(), MLPr_i3(), MLPr_i7(), MLPr_i11()
from plotting import *


def main():
    # %%
    # Generating data
    velocity_init = 0
    velocity_end = 100
    theta_init = 0
    theta_end = 90
    sample_amount = 10000
    boundary_param_fitting = velocity_init, velocity_end, theta_init, theta_end, sample_amount

    print("start generating data...")
    X, y, v, theta, h, r = data_g(*boundary_param_fitting, "RFr_0501.csv")
    data_fitting = X, y, v, theta, h, r

    # %%
    # Setting training parameter
    cv = 5
    scoring = 'r2'
    train_sizes = [1e-3, 1e-2, 1e-1, 1e0]

    # %%
    # Choosing ML model, build up the model
    print("start setting model...")
    h_info, r_info, models, model_paths = RFr(X, y, cv, scoring, train_sizes)

    # %%
    # Plotting learning curves
    print("start plotting learning curves...")
    train_plotting(h_info, r_info)

    # %%
    # Fitting, saving the model
    print("start fitting, saving...")
    model_fitting_saving(models, model_paths, data_fitting)

    # %%
    # 3D plotting (including outlier data)
    v_start = 0
    v_end = 200
    theta_start = 0
    theta_end = 180
    seed_amount = 250
    boundary_param_predict = (v_start, v_end, seed_amount), (theta_start, theta_end, seed_amount)
    print("start predicting, visualizing...")
    data_visualization(boundary_param_predict, models, data_fitting)


if __name__ == "__main__":
    main()

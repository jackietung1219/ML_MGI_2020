import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # projection='3d'
from joblib import dump, load


def train_plotting(h_info, r_info):
    """Plot the training curve"""
    train_sizes_h, train_score_h_mean, test_score_h_mean = h_info
    train_sizes_r, train_score_r_mean, test_score_r_mean = r_info

    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.subplot(1, 2, 1)
    plt.title('Height training')
    plt.plot(train_sizes_h, train_score_h_mean, 'o-', label="Training score")
    plt.plot(train_sizes_h, test_score_h_mean, 'o-', label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("score")
    plt.legend(loc="best")
    # plt.xlim(5, 10000)
    plt.xscale('symlog')
    plt.grid(True)
    plt.gca().xaxis.grid(True, which='minor')
    # plt.ylim(0, 1.05)
    plt.subplot(1, 2, 2)
    plt.title('Distance training')
    plt.plot(train_sizes_r, train_score_r_mean, 'o-', label="Training score")
    plt.plot(train_sizes_r, test_score_r_mean, 'o-', label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("score")
    plt.legend(loc="best")
    # plt.xlim(5, 10000)
    plt.xscale('symlog')
    plt.grid(True)
    plt.gca().xaxis.grid(True, which='minor')
    # plt.ylim(0, 1.05)
    plt.show()


def model_fitting_saving(models, model_paths, data_fitting):
    """Model fitting and saving"""

    model_h, model_r = models
    model_h_path, model_r_path = model_paths
    X, y, v, theta, h, r = data_fitting

    # fitting
    model_h.fit(X, y[:, -2])
    model_r.fit(X, y[:, -1])
    # saving
    dump(model_h, model_h_path)
    dump(model_r, model_r_path)


def data_visualization(boundary_param_predict, models, data_fitting=None):
    """Model predicting and visualizing"""

    v_boundary, theta_boundary = boundary_param_predict
    model_h, model_r = models

    if data_fitting is not None:
        X, y, v, theta, h, r = data_fitting
    else:
        X, y, v, theta, h, r = None, None, None, None, None, None

    v_p = np.random.uniform(*v_boundary)
    theta_p = np.random.uniform(*theta_boundary)

    hp = (v_p * np.sin(theta_p * (np.pi / 180))) ** 2 / 19.6
    rp = (v_p ** 2) * np.sin(2 * theta_p * (np.pi / 180)) / 9.8
    vp_tp = np.stack((v_p, theta_p), axis=-1)

    # data visualizing
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')  # get current axes
    if data_fitting is not None:
        ax.scatter(v[:500], theta[:500], h[:500], c=h[:500], cmap=cm.coolwarm)
    ax.scatter(v_p, theta_p, model_h.predict(vp_tp), c='r', marker='x', label='prediction')
    ax.scatter(v_p, theta_p, hp, c='g', marker='^', label='true')
    ax.text2D(0.05, 0.95, 'height', transform=ax.transAxes)
    ax.set_xlabel('velocity')
    ax.set_ylabel('theta')
    ax.set_zlabel('height')
    ax.legend()
    plt.show()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    if data_fitting is not None:
        ax.scatter(v[:500], theta[:500], r[:500], c=r[:500], cmap=cm.coolwarm)
    ax.scatter(v_p, theta_p, model_r.predict(vp_tp), c='r', marker='x', label='prediction')
    ax.scatter(v_p, theta_p, rp, c='g', marker='^', label='true')
    ax.text2D(0.05, 0.95, 'Distance', transform=ax.transAxes)
    ax.set_xlabel('velocity')
    ax.set_ylabel('theta')
    ax.set_zlabel('distance')
    ax.legend()
    plt.show()

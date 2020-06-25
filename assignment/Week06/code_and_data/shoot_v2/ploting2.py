import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from joblib import dump, load
from sklearn.svm import SVR
from sklearn.manifold import Isomap
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

def data_ploting_parameter(ab, cd, ef, gh, ij):
    vp = np.random.uniform(ab, cd, ij)
    tp = np.random.uniform(ef, gh, ij)
    hp = (vp*np.sin(tp*(np.pi/180)))**2/19.6
    rp = (vp**2)*np.sin(2*tp*(np.pi/180))/9.8
    vptp = np.stack((vp, tp), axis = -1)
    from _tmp import X, y, model_h, model_r, model_h_path, model_r_path
    model_h.fit(X, y[:,-2])
    model_r.fit(X, y[:,-1])
    dump(model_h, model_h_path)
    dump(model_r, model_r_path)
    model_h.predict(vptp)
    model_r.predict(vptp)
    from _tmp import v, theta, h, r
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    ax.scatter(v[:500], theta[:500], h[:500], c=h[:500], cmap=cm.coolwarm)
    ax.scatter(vp, tp, model_h.predict(vptp), c='r',marker='x')
    ax.scatter(vp, tp, hp, c='g',marker='^')
    ax.text2D(0.05, 0.95, 'height', transform=ax.transAxes)
    ax.set_xlabel('velocity')
    ax.set_ylabel('theta')
    ax.set_zlabel('height')
    plt.show()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    ax.scatter(v[:500], theta[:500], r[:500], c=r[:500], cmap=cm.coolwarm)
    ax.scatter(vp, tp, model_r.predict(vptp), c='r',marker='x')
    ax.scatter(vp, tp, rp, c='g',marker='^')
    ax.text2D(0.05, 0.95, 'Distance', transform=ax.transAxes)
    ax.set_xlabel('velocity')
    ax.set_ylabel('theta')
    ax.set_zlabel('distance')
    plt.show()
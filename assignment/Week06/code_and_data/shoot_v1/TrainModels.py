import numpy as np
from joblib import dump, load
from sklearn.svm import SVR
from sklearn.manifold import Isomap
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
import time


def get_model_path_name(es, fs):
    """Get model path"""
    model_h_path = str('./_train_model/' + str(es) + '.joblib')
    model_r_path = str('./_train_model/' + str(fs) + '.joblib')
    return model_h_path, model_r_path


def RFr(X, y, cv, scoring, train_sizes):
    RFr_h = RandomForestRegressor(n_estimators=100, random_state=0)
    RFr_r = RandomForestRegressor(n_estimators=100, random_state=0)
    s1 = time.time()
    h_score = cross_val_score(RFr_h, X, y[:, -2], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('RFr height cross_val:', s2 - s1, 'sec')
    s1 = time.time()
    r_score = cross_val_score(RFr_r, X, y[:, -1], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('RFr distance cross_val:', s2 - s1, 'sec\n')
    print('RFr height test_score:', h_score, '\nRFr height test_score:', np.average(h_score), '\n')
    print('RFr distance test_score:', r_score, '\nRFr distance test_score:', np.average(r_score), '\n')
    s1 = time.time()
    train_sizes_h, train_score_h, test_score_h = learning_curve(RFr_h, X, y[:, -2], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    s2 = time.time()
    print('SVr height learning curve:', s2 - s1, 'sec')
    train_score_h_mean = np.mean(train_score_h, axis=1)
    test_score_h_mean = np.mean(test_score_h, axis=1)
    s1 = time.time()
    train_sizes_r, train_score_r, test_score_r = learning_curve(RFr_r, X, y[:, -1], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    s2 = time.time()
    print('SVr distance learning curve:', s2 - s1, 'sec')
    train_score_r_mean = np.mean(train_score_r, axis=1)
    test_score_r_mean = np.mean(test_score_r, axis=1)

    model_h, model_r = RFr_h, RFr_r
    model_h_path, model_r_path = get_model_path_name('RFr_h', 'RFr_r')

    h_info = (train_sizes_h, train_score_h_mean, test_score_h_mean)
    r_info = (train_sizes_r, train_score_r_mean, test_score_r_mean)

    return h_info, r_info, (model_h, model_r), (model_h_path, model_r_path)


def SVr(X, y, cv, scoring, train_sizes):
    SVr_h = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    SVr_r = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    s1 = time.time()
    h_score = cross_val_score(SVr_h, X, y[:, -2], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('SVr height cross_val:', s2 - s1, 'sec')
    s1 = time.time()
    r_score = cross_val_score(SVr_r, X, y[:, -1], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('SVr distance cross_val:', s2 - s1, 'sec\n')
    print('SVr height test_score:', h_score, '\nSVr height test_score:', np.average(h_score), '\n')
    print('SVr distance test_score:', r_score, '\nSVr distance test_score:', np.average(r_score), '\n')

    s1 = time.time()
    train_sizes_h, train_score_h, test_score_h = learning_curve(SVr_h, X, y[:, -2], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    s2 = time.time()
    print('SVr height learning curve:', s2 - s1, 'sec')
    train_score_h_mean = np.mean(train_score_h, axis=1)
    test_score_h_mean = np.mean(test_score_h, axis=1)
    s1 = time.time()
    train_sizes_r, train_score_r, test_score_r = learning_curve(SVr_r, X, y[:, -1], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    s2 = time.time()
    print('SVr distance learning curve:', s2 - s1, 'sec')
    train_score_r_mean = np.mean(train_score_r, axis=1)
    test_score_r_mean = np.mean(test_score_r, axis=1)

    model_h, model_r = SVr_h, SVr_r
    model_h_path, model_r_path = get_model_path_name('SVr_h', 'SVr_r')

    h_info = (train_sizes_h, train_score_h_mean, test_score_h_mean)
    r_info = (train_sizes_r, train_score_r_mean, test_score_r_mean)

    return h_info, r_info, (model_h, model_r), (model_h_path, model_r_path)


def MLPr_3(X, y, cv, scoring, train_sizes):
    MLPr_3_h = MLPRegressor(hidden_layer_sizes=(1024, 1024, 1024), random_state=0)
    MLPr_3_r = MLPRegressor(hidden_layer_sizes=(1024, 1024, 1024), random_state=0)
    s1 = time.time()
    h_score = cross_val_score(MLPr_3_h, X, y[:, -2], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('MLPr_3_r height cross_val:', s2 - s1, 'sec')
    s1 = time.time()
    r_score = cross_val_score(MLPr_3_r, X, y[:, -1], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('MLPr_3_r distance cross_val:', s2 - s1, 'sec\n')
    print('MLPr_3 height test_score:', h_score, '\nMLPr_3 height test_score:', np.average(h_score), '\n')
    print('MLPr_3 distance test_score:', r_score, '\nMLPr_3 distance test_score:', np.average(r_score), '\n')
    train_sizes_h, train_score_h, test_score_h = learning_curve(MLPr_3_h, X, y[:, -2], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    train_score_h_mean = np.mean(train_score_h, axis=1)
    test_score_h_mean = np.mean(test_score_h, axis=1)
    train_sizes_r, train_score_r, test_score_r = learning_curve(MLPr_3_r, X, y[:, -1], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    train_score_r_mean = np.mean(train_score_r, axis=1)
    test_score_r_mean = np.mean(test_score_r, axis=1)

    model_h, model_r = MLPr_3_h, MLPr_3_r
    model_h_path, model_r_path = get_model_path_name('MLPr_3_h', 'MLPr_3_r')

    h_info = (train_sizes_h, train_score_h_mean, test_score_h_mean)
    r_info = (train_sizes_r, train_score_r_mean, test_score_r_mean)

    return h_info, r_info, (model_h, model_r), (model_h_path, model_r_path)


def MLPr_i3(X, y, cv, scoring, train_sizes):
    # estimator
    MLPr_i3_h = Pipeline([('iso_MLPr_i3_h', Isomap()),
                          ('MLPr_i3_h', MLPRegressor(hidden_layer_sizes=(1024, 1024, 1024), random_state=0))])
    MLPr_i3_r = Pipeline([('iso_MLPr_i3_r', Isomap()),
                          ('MLPr_i3_r', MLPRegressor(hidden_layer_sizes=(1024, 1024, 1024), random_state=0))])

    # cv = the number of folds (5-fold); record the time spent
    s1 = time.time()
    h_score = cross_val_score(MLPr_i3_h, X, y[:, -2], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('MLPr_i3_r height cross_val:', s2 - s1, 'sec')
    s1 = time.time()
    r_score = cross_val_score(MLPr_i3_r, X, y[:, -1], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('MLPr_i3_r distance cross_val:', s2 - s1, 'sec\n')
    print('MLPr_i3_h height test_score:', h_score, '\nMLPr_i3_h height test_score:', np.average(h_score), '\n')
    print('MLPr_i3_r distance test_score:', r_score, '\nMLPr_i3_r distance test_score:', np.average(r_score), '\n')

    train_sizes_h, train_score_h, test_score_h = learning_curve(MLPr_i3_h, X, y[:, -2], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    train_score_h_mean = np.mean(train_score_h, axis=1)
    test_score_h_mean = np.mean(test_score_h, axis=1)
    train_sizes_r, train_score_r, test_score_r = learning_curve(MLPr_i3_r, X, y[:, -1], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    train_score_r_mean = np.mean(train_score_r, axis=1)
    test_score_r_mean = np.mean(test_score_r, axis=1)

    model_h, model_r = MLPr_i3_h, MLPr_i3_r
    model_h_path, model_r_path = get_model_path_name('MLPr_i3_h', 'MLPr_i3_r')

    h_info = (train_sizes_h, train_score_h_mean, test_score_h_mean)
    r_info = (train_sizes_r, train_score_r_mean, test_score_r_mean)

    return h_info, r_info, (model_h, model_r), (model_h_path, model_r_path)


def MLPr_i5(X, y, cv, scoring, train_sizes):
    MLPr_i5_h = Pipeline([('iso_MLPr_i5_h', Isomap()), (
        'MLPr_i5_h', MLPRegressor(hidden_layer_sizes=(1024, 1024, 1024, 1024, 1024), random_state=0))])
    MLPr_i5_r = Pipeline([('iso_MLPr_i5_r', Isomap()), (
        'MLPr_i5_r', MLPRegressor(hidden_layer_sizes=(1024, 1024, 1024, 1024, 1024), random_state=0))])
    s1 = time.time()
    h_score = cross_val_score(MLPr_i5_h, X, y[:, -2], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('MLPr_i5 height cross_val:', s2 - s1, 'sec')
    s1 = time.time()
    r_score = cross_val_score(MLPr_i5_r, X, y[:, -1], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('MLPr_i5 distance cross_val:', s2 - s1, 'sec\n')
    print('MLPr_i5_h height test_score:', h_score, '\nMLPr_i5_h height test_score:', np.average(h_score), '\n')
    print('MLPr_i5_r distance test_score:', r_score, '\nMLPr_i5_r distance test_score:', np.average(r_score), '\n')

    train_sizes_h, train_score_h, test_score_h = learning_curve(MLPr_i5_h, X, y[:, -2], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    train_score_h_mean = np.mean(train_score_h, axis=1)
    test_score_h_mean = np.mean(test_score_h, axis=1)
    train_sizes_r, train_score_r, test_score_r = learning_curve(MLPr_i5_r, X, y[:, -1], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    train_score_r_mean = np.mean(train_score_r, axis=1)
    test_score_r_mean = np.mean(test_score_r, axis=1)

    model_h, model_r = MLPr_i5_h, MLPr_i5_r
    model_h_path, model_r_path = get_model_path_name('MLPr_i5_h', 'MLPr_i5_r')

    h_info = (train_sizes_h, train_score_h_mean, test_score_h_mean)
    r_info = (train_sizes_r, train_score_r_mean, test_score_r_mean)

    return h_info, r_info, (model_h, model_r), (model_h_path, model_r_path)


def MLPr_i7(X, y, cv, scoring, train_sizes):
    MLPr_i7_h = Pipeline([('iso_MLPr_i7_h', Isomap()), (
        'MLPr_i7_h', MLPRegressor(hidden_layer_sizes=(1024, 1024, 1024, 1024, 1024, 1024, 1024), random_state=0))])
    MLPr_i7_r = Pipeline([('iso_MLPr_i7_r', Isomap()), (
        'MLPr_i7_r', MLPRegressor(hidden_layer_sizes=(1024, 1024, 1024, 1024, 1024, 1024, 1024), random_state=0))])
    s1 = time.time()
    h_score = cross_val_score(MLPr_i7_h, X, y[:, -2], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('MLPr_i7 height cross_val:', s2 - s1, 'sec')
    s1 = time.time()
    r_score = cross_val_score(MLPr_i7_r, X, y[:, -1], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('MLPr_i7 distance cross_val:', s2 - s1, 'sec\n')
    print('MLPr_i7_h height test_score:', h_score, '\nMLPr_i7_h height test_score:', np.average(h_score), '\n')
    print('MLPr_i7_r distance test_score:', r_score, '\nMLPr_i7_r distance test_score:', np.average(r_score), '\n')
    train_sizes_h, train_score_h, test_score_h = learning_curve(MLPr_i7_h, X, y[:, -2], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    train_score_h_mean = np.mean(train_score_h, axis=1)
    test_score_h_mean = np.mean(test_score_h, axis=1)
    train_sizes_r, train_score_r, test_score_r = learning_curve(MLPr_i7_r, X, y[:, -1], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    train_score_r_mean = np.mean(train_score_r, axis=1)
    test_score_r_mean = np.mean(test_score_r, axis=1)

    model_h, model_r = MLPr_i7_h, MLPr_i7_r
    model_h_path, model_r_path = get_model_path_name('MLPr_i7_h', 'MLPr_i7_r')

    h_info = (train_sizes_h, train_score_h_mean, test_score_h_mean)
    r_info = (train_sizes_r, train_score_r_mean, test_score_r_mean)

    return h_info, r_info, (model_h, model_r), (model_h_path, model_r_path)


def MLPr_i11(X, y, cv, scoring, train_sizes):
    MLPr_i11_h = Pipeline([('iso_MLPr_i11_h', Isomap()), ('MLPr_i11_h', MLPRegressor(
        hidden_layer_sizes=(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024), random_state=0))])
    MLPr_i11_r = Pipeline([('iso_MLPr_i11_r', Isomap()), ('MLPr_i11_r', MLPRegressor(
        hidden_layer_sizes=(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024), random_state=0))])
    s1 = time.time()
    h_score = cross_val_score(MLPr_i11_h, X, y[:, -2], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('MLPr_i11_r height cross_val:', s2 - s1, 'sec')
    s1 = time.time()
    r_score = cross_val_score(MLPr_i11_r, X, y[:, -1], cv=cv, n_jobs=-1)
    s2 = time.time()
    print('MLPr_i11_r distance cross_val:', s2 - s1, 'sec\n')
    print('MLPr_i11_h height test_score:', h_score, '\nMLPr_i11_h height test_score:', np.average(h_score), '\n')
    print('MLPr_i11_r distance test_score:', r_score, '\nMLPr_i11_r distance test_score:', np.average(r_score), '\n')
    train_sizes_h, train_score_h, test_score_h = learning_curve(MLPr_i11_h, X, y[:, -2], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    train_score_h_mean = np.mean(train_score_h, axis=1)
    test_score_h_mean = np.mean(test_score_h, axis=1)
    train_sizes_r, train_score_r, test_score_r = learning_curve(MLPr_i11_r, X, y[:, -1], cv=cv, scoring=scoring,
                                                                train_sizes=train_sizes)
    train_score_r_mean = np.mean(train_score_r, axis=1)
    test_score_r_mean = np.mean(test_score_r, axis=1)

    model_h, model_r = MLPr_i11_h, MLPr_i11_r
    model_h_path, model_r_path = get_model_path_name('MLPr_i11_h', 'MLPr_i11_r')

    h_info = (train_sizes_h, train_score_h_mean, test_score_h_mean)
    r_info = (train_sizes_r, train_score_r_mean, test_score_r_mean)

    return h_info, r_info, (model_h, model_r), (model_h_path, model_r_path)

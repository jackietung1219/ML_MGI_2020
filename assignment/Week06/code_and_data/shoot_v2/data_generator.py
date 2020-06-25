import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from _tmp import *

def data_g(velocity_init, velocity_end, theta_init, theta_end, sample_amount):
    v = np.random.uniform(velocity_init, velocity_end, sample_amount)
    theta = np.random.uniform(theta_init, theta_end, sample_amount)
    h = (v*np.sin(theta*(np.pi/180)))**2/19.6
    r = (v**2)*np.sin(2*theta*(np.pi/180))/9.8
    X = np.stack((v, theta), axis = -1)
    y = np.stack((h, r), axis = -1)
    data = np.stack((v, theta, h, r), axis = -1)
    data_time = datetime.datetime.now()
    np.savetxt('data_{}-{}-{}_{}-{}-{}.csv'.format(data_time.year, data_time.month, data_time.day, data_time.hour, data_time.minute, data_time.second), data, delimiter=",")
    data_set(X, y, v, theta, h, r)
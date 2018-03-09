#!/usr/bin/env python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from ML_ex01.Classification.utils import PCA, split_set, normalize_data, join_min_max

# Load training and testing data
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]

X_train, X_test = normalize_data(X_train, X_test)

def get_predictions_SGD(X_train, y_train, X_test, learning_rate=0.005, iterations=40000):
    X_train = np.concatenate((np.ones((X_train.shape[0],1)),X_train),axis=1)
    X_test = np.concatenate((np.ones((X_test.shape[0],1)),X_test),axis=1)

    weights = np.ones(X_train.shape[1])
    sample_count = X_train.shape[0]

    for i in range(iterations):
        prediction = X_train.dot(weights)
        error = y_train - prediction
        print("{} \terror: {}".format(i, np.mean(error.dot(X_train))))
        weights += (learning_rate/sample_count) * (error.dot(X_train) + 0.01/sample_count * weights)

    return X_test.dot(weights)


def get_predictions(X_train, y_train, X_test):
    X_train = np.concatenate((np.ones((X_train.shape[0],1)),X_train),axis=1)
    X_test = np.concatenate((np.ones((X_test.shape[0],1)),X_test),axis=1)

    weights = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

    return X_test.dot(weights)

def score(X_val, y_val, X_train, y_train):
    return 1 - sum((get_predictions_SGD(X_train,y_train, X_val) - y_val)**2) / sum((y_val - np.mean(y_val))**2)


X_train = np.delete(X_train, [2,5], axis=1)
X_test = np.delete(X_test, [2,5], axis=1)


dims = X_train.shape[1]
for dimension in range(dims):
    X_train = np.hstack((X_train,np.reshape((X_train[:,dimension]**2), (X_train.shape[0], 1))))
    X_test = np.hstack((X_test,np.reshape((X_test[:,dimension]**2), (X_test.shape[0], 1))))
    # X_train = np.hstack((X_train, np.reshape((X_train[:, dimension] ** 3), (X_train.shape[0], 1))))
    # X_test = np.hstack((X_test, np.reshape((X_test[:, dimension] ** 3), (X_test.shape[0], 1))))


# X_train, X_test = join_min_max(X_train, X_test)
# X_train, X_test = normalize_data(X_train, X_test)

X_temp, y_temp, X_val, Y_val = split_set(X_train, y_train, 40)

# print(score(X_val, Y_val, X_temp, y_temp))

y_pred = get_predictions_SGD(X_train,y_train,X_test)

# Fit model and predict test values
# y_pred = np.random.randint(y_train.min(), y_train.max(), X_test.shape[0])


# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission_reg_.csv', y_pred_pp, fmt='%d,%f', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.

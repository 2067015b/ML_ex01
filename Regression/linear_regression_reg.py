#!/usr/bin/env python
import math
import random
import numpy as np

# ========================================== HELPER FUNCTIONS ==========================================================

def split_set(x, y, val_size):
    """ Split the given training set into a validation and training set. """
    val_indices = random.sample(range(len(y)), k=val_size)

    X_val = x[val_indices, :]
    Y_val = y[val_indices]

    y_temp = np.delete(y, val_indices, axis=0)
    X_temp = np.delete(x, val_indices, axis=0)

    return X_temp, y_temp, X_val, Y_val

def normalize_data(X_train, X_test):
    """ Normalize all feature values with respect to its maximum """
    maximums_1 = X_train.max(axis=0)
    maximums_2 = X_test.max(axis=0)
    maximums = np.stack((maximums_1, maximums_2), axis=1).max(axis=1)
    return X_train / maximums, X_test / maximums

def get_predictions_SGD_l2(X_train, y_train, X_test, learning_rate=0.005, iterations=27500):
    """ Gradient descent model for weights estimation that uses L2 regularization """
    X_train = np.concatenate((np.ones((X_train.shape[0],1)),X_train),axis=1)
    X_test = np.concatenate((np.ones((X_test.shape[0],1)),X_test),axis=1)

    weights = np.ones(X_train.shape[1])
    sample_count = X_train.shape[0]

    for i in range(iterations):
        prediction = X_train.dot(weights)
        error = y_train - prediction
        if (i % 500 == 0):
            print("{} \terror: {}".format(i, np.mean(error.dot(X_train))))
        weights += (learning_rate/sample_count) * (error.dot(X_train) + 12/sample_count * weights)

    return X_test.dot(weights)

def get_predictions_SGD_l1(X_train, y_train, X_test, learning_rate=0.005, iterations=41000):
    """ Gradient descent model for weights estimation that uses L1 regularization """
    X_train = np.concatenate((np.ones((X_train.shape[0],1)),X_train),axis=1)
    X_test = np.concatenate((np.ones((X_test.shape[0],1)),X_test),axis=1)

    weights = np.ones(X_train.shape[1])
    sample_count = X_train.shape[0]

    for i in range(iterations):
        prediction = X_train.dot(weights)
        error = y_train - prediction
        if (i % 500 == 0):
            print("{} \terror: {}".format(i, np.mean(error.dot(X_train))))
        weights += (learning_rate/sample_count) * (error.dot(X_train) + 12/sample_count * np.sign(weights))

    return X_test.dot(weights)

def score(X_val, y_val, X_train, y_train, method, args=None):
    """ RMSE scoring function """
    return math.sqrt(1 / y_val.shape[0] * sum((y_val - method(X_train, y_train, X_val,*args)) ** 2))

# ========================================== CODE ======================================================================

# Set if the model is being evaluated
evaluate = False
VAL_EPOCHS = 15

# Load training and testing data
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]

X_train, X_test = normalize_data(X_train, X_test)

# Delete the noisy features (defined by doing an ablation study)
X_train = np.delete(X_train, [2,5], axis=1)
X_test = np.delete(X_test, [2,5], axis=1)

# Add the inv(x) feature for the first feature
X_train = np.hstack((X_train,np.reshape(1/(X_train[:,0]+1), (X_train.shape[0], 1))))
X_test = np.hstack((X_test,np.reshape((1/(X_test[:,0]+1)), (X_test.shape[0], 1))))

# Add the 2nd polynomial features generated from the remaining features
dims = X_train.shape[1]
for dimension in range(1,dims-1):
    X_train = np.hstack((X_train,np.reshape(X_train[:,dimension]**2, (X_train.shape[0], 1))))
    X_test = np.hstack((X_test,np.reshape((X_test[:,dimension])**2, (X_test.shape[0], 1))))

# Normalize the data
X_train, X_test = normalize_data(X_train, X_test)

# If we are performing cross-validation
if evaluate:
    result = 0
    for i in range(VAL_EPOCHS):
        print(i)
        X_temp, y_temp, X_val, Y_val = split_set(X_train, y_train, 17)
        result += score(X_val, Y_val, X_temp, y_temp, get_predictions_SGD_l1, (0.005, 41000))
    print(result/VAL_EPOCHS)

# Fit the model and get predictions
y_pred = get_predictions_SGD_l1(X_train,y_train,X_test)

# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission_reg_l1.csv', y_pred_pp, fmt='%d,%f', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.

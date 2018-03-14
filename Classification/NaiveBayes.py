#!/usr/bin/env python

import numpy as np

from Classification.utils import PCA, LDA, normalize_data, split_set


FEATURES_TO_KEEP = [24, 26, 25, 27, 36, 0, 1, 2, 11, 6, 8, 18, 45, 47, 58, 60, 102, 111, 110]

# Return the percentage of correct predictions
def evaluate(predictions, gt):
    correct = 0
    for val_0, val_1 in zip(predictions, gt):
        if val_0 == val_1:
            correct += 1

    return correct / float(len(gt))


# Fit model and predict test values using the Naive Bayes Gaussian model
def get_predictions(X_train, y_train, X_test):

    sorted = [[x for x, label in zip(X_train, y_train) if label == cls] for cls in [1,2]]
    model = np.array([np.c_[np.mean(cls, axis=0), np.std(cls, axis=0)]
                      for cls in sorted])

    log_probabilities = []
    for data_point in X_test:
        class_probability = []
        for cls in model:
            sum = 0
            for param, feature in zip(cls, data_point):
                exponent = np.exp(- ((feature - param[0]) ** 2 / (2 * param[1] ** 2)))
                gaussian_prob = np.log(exponent / (np.sqrt(2 * np.pi) * param[1]))
                sum += gaussian_prob
            class_probability.append(sum)
        log_probabilities.append(class_probability)

    return np.argmax(np.array(log_probabilities), axis=1) + 1  # Classes are indexed from 1 while arrays start at 0


# Load training and testing data
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

X_train = X_train[:, FEATURES_TO_KEEP]
X_test = X_test[:, FEATURES_TO_KEEP]

prior_mean = np.mean(X_train, axis=0)
prior_std = np.std(X_train, axis=0)

delete = []
for data_point in range(X_train.shape[0]):
    for feature in range(X_train.shape[1]):
        if abs(X_train[data_point,feature] - prior_mean[feature]) > prior_std[feature]*3:
            delete.append(data_point)
            break

print(len(delete))
X_train = np.delete(X_train,delete, axis=0)
y_train = np.delete(y_train,delete, axis=0)

X_train, X_test = normalize_data(X_train, X_test)

reduced = LDA([X_train, X_test], y_train, X_train.shape[1], 13)
X_train, X_test = np.real(reduced[0]), np.real(reduced[1])

score = 0.0
for i in range(1000):
    X_temp, y_temp, X_val, Y_val = split_set(X_train, y_train, 120)

    y_pred = get_predictions(X_temp, y_temp, X_val)
    value = evaluate(y_pred,Y_val)
    score += value
    if i%100 == 0:
        print("intermediate: {}".format(value))

print("final: {}".format(score / 1000))

y_pred = get_predictions(X_train,y_train,X_test)

# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.

test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
file_name = 'submissions/my_submission_NB_sel_.csv'
np.savetxt(file_name, y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.

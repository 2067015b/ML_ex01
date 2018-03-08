#!/usr/bin/env python

import numpy as np
import random

# Initial script with Naive Bayes and an automatically determined set of features removed

AVERAGING_FACTOR = 5
FEATURES_TO_REMOVE = 30
VALIDATION_SPLIT = 60


# Return the percentage of correct predictions
def evaluate(predictions, gt):
    correct = 0
    for val_0, val_1 in zip(predictions, gt):
        if val_0 == val_1:
            correct += 1

    return correct / float(len(gt))


# Fit model and predict test values using the Naive Bayes Gaussian model
def get_predictions(X_train, y_train, X_test):
    sorted = [[x for x, t in zip(X_train, y_train) if t == c] for c in np.unique(y_train)]
    model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)]
                      for i in sorted])

    log_probabilities = []
    for data_point in X_test:
        class_probability = []
        for class_ in model:
            sum = 0
            for param, feature in zip(class_, data_point):
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

# Number of features to remove
for i in range(FEATURES_TO_REMOVE):

    worst_feature = None
    best_performance = 0.0

    for feature in range(X_train.shape[1]):
        # Delete one feature at a time
        X_temp_ = np.delete(X_train, feature, axis=1)

        performance = 0
        for j in range(AVERAGING_FACTOR):
            # Split the set into training and validation
            val_indices = random.sample(range(len(y_train)), k=VALIDATION_SPLIT)

            X_val = X_temp_[val_indices, :]
            Y_val = y_train[val_indices]

            y_temp = np.delete(y_train, val_indices, axis=0)
            X_temp = np.delete(X_temp_, val_indices, axis=0)

            # Get predictions using the set of selected features
            pred = get_predictions(X_temp, y_temp, X_val)

            # Get evaluation metrics from the validation split
            performance += evaluate(pred, Y_val)

        # Compare performance against models using different set of features
        performance /= AVERAGING_FACTOR
        if performance > best_performance:
            best_performance = performance
            worst_feature = feature

    print("{} feature(s) removed - best performance: {}, feature to delete: {}".format(i + 1, best_performance,
                                                                                       worst_feature))

    # Remove the selected feature from the final training and test sets
    if worst_feature:
        X_train = np.delete(X_train, worst_feature, axis=1)
        X_test = np.delete(X_test, worst_feature, axis=1)

    y_pred = get_predictions(X_train, y_train, X_test)

    # Arrange answer in two columns. First column (with header "Id") is an
    # enumeration from 0 to n-1, where n is the number of test points. Second
    # column (with header "EpiOrStroma" is the predictions.

    test_header = "Id,EpiOrStroma"
    n_points = X_test.shape[0]
    y_pred_pp = np.ones((n_points, 2))
    y_pred_pp[:, 0] = range(n_points)
    y_pred_pp[:, 1] = y_pred
    file_name = 'submissions/my_submission_{}.csv'.format(i)
    np.savetxt(file_name, y_pred_pp, fmt='%d', delimiter=",",
               header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.

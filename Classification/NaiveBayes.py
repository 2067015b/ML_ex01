#!/usr/bin/env python
import random

import numpy as np

FEATURES_TO_KEEP = [24, 26, 25, 27, 36, 0, 1, 2, 11, 6, 8, 18, 45, 47, 58, 60, 102, 111, 110]

# =========================================== HELPER FUNCTIONS =========================================================

def LDA(dataset, y, initial_dim, target_dim):

    # Split the training dataset given the two classes
    data = {0: [], 1: []}
    for i in range(y.shape[0]):
        if y[i] == 2:
            data[1].append(dataset[0][i, :])
        else:
            data[0].append(dataset[0][i, :])

    data[0] = np.array(data[0])
    data[1] = np.array(data[1])

    # Calculate mean feature values per each class
    class_mean = []
    for cls in range(2):
        class_mean.append(np.mean(data[cls], axis=0))

    # Compute the within class deviation from the mean
    within_class_matrix = np.zeros((initial_dim, initial_dim))
    for cls, mean_value in zip(range(2), class_mean):
        diff_accum_matrix = np.zeros((initial_dim, initial_dim))
        for row in data[cls]:
            row, mean_value = row.reshape(initial_dim, 1), mean_value.reshape(initial_dim, 1)
            diff_accum_matrix += (row - mean_value).dot((row - mean_value).T)
        within_class_matrix += diff_accum_matrix

    # Compute the deviation of each class from the overall mean
    total_mean = np.mean(dataset[0], axis=0)
    total_mean = total_mean.reshape(initial_dim, 1)

    between_class_matrix = np.zeros((initial_dim, initial_dim))
    for i, mean_vector in enumerate(class_mean):
        n = data[i].shape[0]
        mean_vector = mean_vector.reshape(initial_dim, 1)
        between_class_matrix += n * (mean_vector - total_mean).dot((mean_vector - total_mean).T)

    eig_values, eig_vectors = np.linalg.eig(np.linalg.inv(within_class_matrix).dot(between_class_matrix))

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_tuples = [(np.abs(eig_values[i]), eig_vectors[:, i]) for i in range(len(eig_values))]

    # Sort the (eigenvalue, eigenvector) tuples
    eig_tuples = sorted(eig_tuples, key=lambda k: k[0], reverse=True)

    weight_matrix = eig_tuples[0][1].reshape(initial_dim, 1)
    for i in range(1,target_dim):
        weight_matrix = np.hstack((weight_matrix, eig_tuples[i][1].reshape(initial_dim, 1)))

    result = []
    for set in dataset:
        result.append(set.dot(weight_matrix))

    return result

def split_set(x, y, val_size):
    val_indices = random.sample(range(len(y)), k=val_size)

    X_val = x[val_indices, :]
    Y_val = y[val_indices]

    y_temp = np.delete(y, val_indices, axis=0)
    X_temp = np.delete(x, val_indices, axis=0)

    return X_temp, y_temp, X_val, Y_val


def normalize_data(X_train, X_test):
    maximums_1 = X_train.max(axis=0)
    maximums_2 = X_test.max(axis=0)
    maximums = np.stack((maximums_1, maximums_2), axis=1).max(axis=1)
    return X_train / maximums, X_test / maximums


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


# ================================================== CODE ==============================================================

test_model = True
VAL_ITER = 1000

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

X_train = np.delete(X_train,delete, axis=0)
y_train = np.delete(y_train,delete, axis=0)

X_train, X_test = normalize_data(X_train, X_test)

reduced = LDA([X_train, X_test], y_train, X_train.shape[1], 13)
X_train, X_test = np.real(reduced[0]), np.real(reduced[1])

if test_model:
    score = 0.0
    for i in range(VAL_ITER):
        X_temp, y_temp, X_val, Y_val = split_set(X_train, y_train, 60)

        y_pred = get_predictions(X_temp, y_temp, X_val)
        value = evaluate(y_pred,Y_val)
        score += value
        if i%100 == 0:
            print("intermediate: {}".format(value))

    print("final: {}".format(score / VAL_ITER))

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

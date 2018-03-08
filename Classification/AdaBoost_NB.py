#!/usr/bin/env python

import numpy as np
import random

from Classification.utils import split_set

AVERAGING_FACTOR = 5
FEATURES_TO_REMOVE = 30
VALIDATION_SPLIT = 60
NO_OF_CLASSIFIERS = 600
TREE_BIAS = 0.1
VOTE_SCALING_FACTOR = (1+TREE_BIAS*VALIDATION_SPLIT)*NO_OF_CLASSIFIERS
REUSE = False

new_instance = False


class Model:
    def __init__(self, features_to_delete=None):
        self.features_to_delete = features_to_delete

    # Fit model and predict test values using the Naive Bayes Gaussian model
    def get_predictions(self, X_train_orig, y_train, X_test_orig):

        if self.features_to_delete:
            X_train = np.delete(X_train_orig, self.features_to_delete, axis=1)
            X_test = np.delete(X_test_orig, self.features_to_delete, axis=1)
        else:
            X_train = X_train_orig
            X_test = X_test_orig

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


# Return the percentage of correct predictions
def evaluate(predictions, gt):
    correct = 0
    for val_0, val_1 in zip(predictions, gt):
        if val_0 == val_1:
            correct += 1

    return correct / float(len(gt))

if REUSE:
    X_train = np.loadtxt('X_train_mod.csv', delimiter=',')
    X_test = np.loadtxt('X_test_mod.csv', delimiter=',')
    y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]
    new_instance = False

else:
    # Load training and testing data
    X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
    X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
    y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

if new_instance:
    # Number of features to remove
    for i in range(FEATURES_TO_REMOVE):

        best_feature = None
        best_performance = 0.0
        possible_best = None

        for feature in range(X_train.shape[1]):
            # Delete one feature at a time
            model = Model(feature)

            performance = 0
            for j in range(AVERAGING_FACTOR):
                # Split the set into training and validation
                val_indices = random.sample(range(len(y_train)), k=VALIDATION_SPLIT)

                X_val = X_train[val_indices, :]
                Y_val = y_train[val_indices]

                y_temp = np.delete(y_train, val_indices, axis=0)
                X_temp = np.delete(X_train, val_indices, axis=0)

                # Get predictions using the set of selected features
                pred = model.get_predictions(X_temp, y_temp, X_val)

                # Get evaluation metrics from the validation split
                performance += evaluate(pred, Y_val)

            # Compare performance against models using different set of features
            performance /= AVERAGING_FACTOR
            if performance > best_performance:
                best_performance = performance
                best_feature = feature

        print("{} feature(s) removed - best performance: {}, feature to delete: {}".format(i + 1, best_performance,
                                                                                           best_feature))

        # Remove the selected feature from the final training and test sets
        if best_feature:
            X_train = np.delete(X_train, best_feature, axis=1)
            X_test = np.delete(X_test, best_feature, axis=1)
            if float(i) >= (FEATURES_TO_REMOVE*0.7) and (not possible_best or best_performance >= possible_best):
                possible_best = best_performance
                np.savetxt("X_train_mod.csv", X_train, fmt='%f', delimiter=",", comments="")
                np.savetxt("X_test_mod.csv", X_test, fmt='%f', delimiter=",", comments="")



forest = [Model()]
features_to_del = [0] + random.choices(range(X_test.shape[1]), k=NO_OF_CLASSIFIERS - 1)
for feature_count in features_to_del:
    feature_set = random.sample(range(X_test.shape[1]), k=feature_count)
    forest.append(Model(feature_set))

print("Generated the forest.")

votes_1 = np.zeros((X_test.shape[0],))
votes_2 = np.zeros((X_test.shape[0],))
for j, tree in enumerate(forest):
    tree_weight = 1

    for trial in range(AVERAGING_FACTOR*2):
        # Split the set into training and validation
        X_temp, y_temp, X_val, Y_val = split_set(X_train, y_train, 120)

        train_pred = tree.get_predictions(X_temp, y_temp, X_val)

        # Calculate the weight of this classifier
        for gt, p in zip(Y_val, train_pred):
            if gt == p:
                tree_weight += (TREE_BIAS/(AVERAGING_FACTOR*2))/VALIDATION_SPLIT
            else:
                tree_weight -= (TREE_BIAS/(AVERAGING_FACTOR))/VALIDATION_SPLIT # not AVERAGING_FACTOR *2 as we want to further penalize bad guesses

    # Given the weight of this classifier, cast votes to the corresponding class
    pred = tree.get_predictions(X_train, y_train, X_test)

    for i, y in enumerate(pred):
        if y == 1:
            current = votes_1[i]
            votes_1[i] = current + tree_weight

        else:
            current = votes_2[i]
            votes_2[i] = current + tree_weight
    print("Tree no. {}: \n\t1: {} \n\t2: {}".format(j, votes_1, votes_2))
    # print("Got votes from classifier {}.".format(j))

votes = np.stack((votes_1, votes_2), axis=1)/VOTE_SCALING_FACTOR
y_pred = list(np.argmax(votes, axis=1) + 1)


# y_pred = get_predictions(X_train, y_train, X_test)

# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.

test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
file_name = 'submissions/my_submission_ADA_{}_a.csv'.format(NO_OF_CLASSIFIERS)
np.savetxt(file_name, y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.

#!/usr/bin/env python

import numpy as np
import operator
import math
import pylab as plt


# Load training and testing data
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]


def euclideanDistance(instance1, instance2):
    """
        Calculates the distance between 2 points by implementing the Euclidean Algorithm
        @param instance1 - point 1
        @param instance2 - point 2

        @return distance
    """

    distance = 0
    max_index = min(len(instance1), len(instance2))
    for x in range(max_index):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def findNeighbours(training_set, test_instance, k):
    """
        Finds the k nearest neighbours.
        @param training_set - training data
        @param test_instance - the data point being looked up
        @param k - amount of neighbours to return

        @return neighbours - list of k nearest neighbours
    """

    distances = []

    for x in range(len(training_set)):
        # Append data point and equivalent distance
        distances.append((training_set[x], euclideanDistance(test_instance, training_set[x])))

    # sort distances to find the closest (head of list)
    distances.sort(key=operator.itemgetter(1))

    # append the k-closest neighbours
    neighbours = []
    for x in range(k):
        neighbours.append(distances[x][0])

    return neighbours


def castVote(neighbours):
    """
        Reads classification for the neighbours and based on the majority, predicts the classification of the data point being tested
        @param neighbours - list of neighbours
        @return 1 for epithelial, 2 for stromal
    """

    votes = [0, 0]

    for x in range(len(neighbours)):
        response = neighbours[x][0]
        if response == 1.0:
            votes[0] += 1
        else:
            votes[1] += 1

    # List index starts in 0. Adds 1 to return 1 or 2.
    return votes.index(max(votes)) + 1


def predictResult(training_set, test_instance, k):
    """
        Calls the necessary functions in order to predict the result.
        @param training_set - training data
        @param test_instance - the data point being looked up
        @param k - amount of neighbours to return

        @return predicted result
    """
    neigbours = (findNeighbours(training_set, test_instance, k))
    print (neigbours)
    return castVote(neigbours)


def knn(neighbours, data, data_train, data_test):
    """
        Executes the code.
    """

    correct = incorrect = 0.0

    for index in range(len(data -1)):
        print(
        "*"),  # loading bar

        x = predictResult(data_train, data_test[index], neighbours)
        if (x == data[index]):
            correct += 1
        else:
            incorrect += 1

    print ("\n correct", correct, 'incorrect', incorrect, 'acuracy:', (correct / (correct + incorrect)) * 100)
    return correct, incorrect, (correct / (correct + incorrect)) * 100

print ((y_train[599]))
knn(10, y_train, X_test, X_train)
# Fit model and predict test values
y_pred = np.random.randint(1, 3, X_test.shape[0])

# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.
    
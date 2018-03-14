import random

import numpy as np

# ============================================= HELPER FUNCTIONS ==================================================

def normalize_data(X_train, X_test):
    """ Normalize all feature values with respect to its maximum """
    maximums_1 = X_train.max(axis=0)
    maximums_2 = X_test.max(axis=0)
    maximums = np.stack((maximums_1, maximums_2), axis=1).max(axis=1)
    return X_train / maximums, X_test / maximums

def split_set(x, y, val_size):
    """ Split the given training set into a validation and training set. """
    val_indices = random.sample(range(len(y)), k=val_size)

    X_val = x[val_indices, :]
    Y_val = y[val_indices]

    y_temp = np.delete(y, val_indices, axis=0)
    X_temp = np.delete(x, val_indices, axis=0)

    return X_temp, y_temp, X_val, Y_val

def evaluate(predictions, gt):
    """ Evaluation function that returns the percentage of correct predictions """
    correct = 0
    for val_0, val_1 in zip(predictions, gt):
        if val_0 == val_1:
            correct += 1

    return correct / float(len(gt))

def predict(datapoint, weights):
    """ Sigmoid activation function """
    pred = 0
    for i in range(len(datapoint)):
        pred += weights[i] * datapoint[i]
    return 1.0 / (1.0 + np.exp(-pred))

def train_model(X_train, y_train, l_rate=0.5, n_epoch=50000):
    """ Train the model using gradient descent """

    # Add the intercept
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)

    # Initialize weights
    weights = [0.5 for i in range(X_train.shape[1])]
    for epoch in range(n_epoch):
        sum_error = 0
        for i in range(X_train.shape[0]):
            # Get the predicted value for this datapoint
            pred = predict(X_train[i,:], weights)
            # Calculate the error
            error = y_train[i] - pred
            # For monitoring purposes
            sum_error += error ** 2
            # Update the weights
            for j in range(len(weights)):
                weights[j] = weights[j] + l_rate * error * pred * (1.0 - pred) * X_train[i,j]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights

# Train the model using SGD and a L2 regularization
def train_model_l2(X_train, y_train, l_rate=0.1, n_epoch=5000, lam=0.001, decay=0.999):
    """ Function to rain the model using SGD and L2 regularization """
    # Add the intercept
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
    # Initialize the weights
    weights = [0.5 for i in range(X_train.shape[1])]

    # m = total number of training datapoints
    m = X_train.shape[1]
    for epoch in range(n_epoch):
        sum_error = 0
        for i in range(len(weights)):
            sum_gradient = 0
            for j in range(X_train.shape[0]):
                pred = predict(X_train[j,:], weights)
                error = pred - y_train[j]
                # For monitoring purposes
                sum_error += error ** 2
                sum_gradient += error * X_train[j,i]
            # No need to regularize the intercept
            if i==0:
                weights[i] = weights[i] - l_rate/m * sum_gradient
            else:
                weights[i] = weights[i] - l_rate * (sum_gradient/m + lam/m * weights[i])

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error/len(weights)))
        if (epoch + 1 % 100):
            l_rate *= decay
    return weights

def train_model_l1(X_train, y_train, l_rate=0.1, n_epoch=5000, lam=0.001, decay=0.999):
    """ Function to rain the model using SGD and L1 regularization """
    # Add the intercept
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
    # Initialize the weights
    weights = [0.5 for i in range(X_train.shape[1])]

    # m = total number of training datapoints
    m = X_train.shape[1]
    for epoch in range(n_epoch):
        sum_error = 0
        for i in range(len(weights)):
            sum_gradient = 0
            for j in range(X_train.shape[0]):
                pred = predict(X_train[j,:], weights)
                error = pred - y_train[j]
                # For monitoring purposes
                sum_error += error ** 2
                sum_gradient += error * X_train[j,i]

            # Do not regularize the intercept
            if i==0:
                weights[i] = weights[i] - l_rate/m * sum_gradient
            else:
                # The L1 regularization parameter is an absolute value, hence the derivation is the sign of this value
                if weights[i] >= 0:
                    weights[i] = weights[i] - l_rate * (sum_gradient/m + lam/m)
                else:
                    weights[i] = weights[i] - l_rate * (sum_gradient/m - lam/m)


        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error/len(weights)))
        if (epoch + 1 % 100):
            l_rate *= decay
    return weights

# ============================================= CODE =============================================================

test_model = False
VAL_ITER = 3

# Load testing and training data
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]


# Normalize the feature values
X_train, X_test = normalize_data(X_train, X_test)

# Sigmoid classification is in range of 0 to 1
y_train -= 1

# Add intercept to the test data
X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

if test_model:
    score = 0.0
    for i in range(VAL_ITER):
        X_temp, y_temp, X_val, Y_val = split_set(X_train, y_train, 60)

        # Get the trained weights of the model
        weights = train_model_l1(X_temp, y_temp)

        # Get predictions for the test data
        y_val = []
        for row in X_val:
            y_val.append(predict(row, weights))

        value = evaluate(y_val, Y_val)
        score += value
        if i % 100 == 0:
            print("intermediate: {}".format(value))

    print("final: {}".format(score / VAL_ITER))


# Get the trained weights of the model
weights = train_model_l1(X_train, y_train)

# Get predictions for the test data
y_pred = []
for row in X_test:
    y_pred.append(predict(row,weights))

# Labels are in the range 0-1
y_pred = np.round(np.array(y_pred))
y_pred += 1

test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
file_name = 'my_submission_log_reg_mod.csv'
np.savetxt(file_name, y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")

import numpy as np

from ML_ex01.CONFIG import FEATURES_TO_KEEP
from ML_ex01.Classification.utils import PCA, normalize_data, LDA

# Load training and testing data
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

# ====================================== METHOD 1 ===============================================

def prediction(scores):
    return 1 / (1 + np.exp(-scores))
    # return .5 * (1 + np.tanh(.5 * scores))


def log_likelihood(x, y, weights):
    scores = np.dot(x, weights)
    ll = np.sum(y * scores - np.log(1 + np.exp(scores)))
    return ll


def logistic_regression(X_train, y_train, iterations=65000, learning_rate=0.001, add_intercept=True):
    if add_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))

    weights = np.ones(X_train.shape[1])

    for step in range(iterations):
        scores = np.dot(X_train, weights)
        predictions = prediction(scores)

        additional_par = (1.0 - predictions)* predictions
        # Update weights with gradient
        output_error_signal = np.square(y_train - predictions) * additional_par

        gradient = np.dot(X_train.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 1000 == 0:
            print("error: {}".format(np.mean(np.abs(output_error_signal))))
            # print(log_likelihood(X_train, y_train, weights))

    return weights

# ====================================== METHOD 2 ===============================================

# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = 0
    for i in range(len(row) - 1):
        yhat += coefficients[i] * row[i]
    # return .5 * (1 + np.tanh(.5 * yhat))
    return 1.0 / (1.0 + np.exp(-yhat))


# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate=0.001, n_epoch=3800, lam=0.001, decay=0.9999):
    coef = [0.5 for i in range(train.shape[1])]
    m = train.shape[1]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            # print(yhat)
            error = yhat - row[-1]
            sum_error += error ** 2
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(1,len(row)):
                # coef[i] = coef[i] + l_rate * (error * yhat * (1.0 - yhat) * row[i-1] + lam * coef[i])
                coef[i] = coef[i] - l_rate * (error * row[i-1] + lam * coef[i])

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        if (epoch + 1 % 100):
            l_rate *= decay
    return coef

def coefficients_sgd_l2(train, l_rate=0.1, n_epoch=3800, lam=0.001, decay=0.999):
    coef = [0.5 for i in range(train.shape[1])]
    intercept = np.ones((train.shape[0], 1))
    train = np.hstack((intercept, train))
    m = train.shape[1]
    for epoch in range(n_epoch):
        sum_error = 0
        for i in range(len(coef)):
            sum_gradient = 0
            for row in train:
                yhat = predict(row, coef)
                error = yhat - row[-1]
                sum_error += error ** 2
                sum_gradient += error * row[i]
            if i==0:
                coef[i] = coef[i] - l_rate/m * sum_gradient
            else:
                coef[i] = coef[i] - l_rate * (sum_gradient/m + lam/m * coef[i])

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error/len(coef)))
        if (epoch + 1 % 100):
            l_rate *= decay
    return coef

def coefficients_sgd_l1(train, l_rate=0.1, n_epoch=3800, lam=0.001, decay=0.999):
    coef = [0.5 for i in range(train.shape[1])]
    intercept = np.ones((train.shape[0], 1))
    train = np.hstack((intercept, train))
    m = train.shape[1]
    for epoch in range(n_epoch):
        sum_error = 0
        for i in range(len(coef)):
            sum_gradient = 0
            for row in train:
                yhat = predict(row, coef)
                error = yhat - row[-1]
                sum_error += error ** 2
                sum_gradient += error * row[i]
            if i==0:
                coef[i] = coef[i] - l_rate/m * sum_gradient
            else:
                if coef[i] >= 0:
                    coef[i] = coef[i] - l_rate * (sum_gradient/m + lam/m)
                else:
                    coef[i] = coef[i] - l_rate * (sum_gradient/m - lam/m)


        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error/len(coef)))
        if (epoch + 1 % 100):
            l_rate *= decay
    return coef

# ================================================================================================

X_train = X_train[:, FEATURES_TO_KEEP]
X_test = X_test[:, FEATURES_TO_KEEP]
X_train, X_test = normalize_data(X_train, X_test)

# reduced = LDA([X_train, X_test], y_train, X_train.shape[1], 13)
# X_train, X_test = np.real(reduced[0]), np.real(reduced[1])
# X_train, X_test = normalize_data(X_train, X_test)

y_train -= 1

X_train = np.hstack((X_train,np.reshape(y_train,(y_train.shape[0],1))))
weights = coefficients_sgd_l2(X_train)

y_pred = []
for row in X_test:
    y_pred.append(predict(row,weights))

# intercept = np.ones((X_test.shape[0], 1))
# X_test = np.hstack((intercept, X_test))
# weights = logistic_regression(X_train,y_train)

# y_pred = np.round(prediction(np.dot(X_test,weights)))
y_pred = np.round(y_pred) + 1
print(y_pred)

test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
file_name = 'submissions/my_submission_log_reg_.csv'
np.savetxt(file_name, y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")
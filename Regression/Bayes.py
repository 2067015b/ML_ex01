import numpy as np
import matplotlib.pyplot as plt

# Load training and testing data
from Classification.utils import PCA, join_min_max, split_set, remove_outliers

X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]

def score(X_val, y_val, X_train, y_train):
    return 1 - sum((get_bayes_prediction(X_train,y_train, X_val) - y_val)**2) / sum((y_val - np.mean(y_val))**2)

def get_bayes_prediction(X_temp, y_temp, X_val):
    # prior mean
    prior_mean = np.mean(X_temp, axis=0)
    prior_std = np.std(X_temp, axis=0)
    prior_cov = prior_std * np.eye(prior_std.shape[0])

    XX = np.dot(X_temp.T, X_temp)
    XXinv = np.linalg.inv(XX)
    Xt = np.dot(X_temp.T, y_temp)
    w = np.dot(XXinv, Xt)

    tXw = y_temp - np.dot(X_temp, w)
    sig_sq = np.dot(tXw.T, tXw) / len(y_temp)

    posterior_cov = np.linalg.inv((1.0/sig_sq)*np.dot(X_temp.T,X_temp) + np.linalg.inv(prior_cov))
    posterior_mu = ((1.0/sig_sq))*np.dot(posterior_cov,np.dot(X_temp.T,y_temp))


    n_samps = 20000
    w_samples = np.random.multivariate_normal(posterior_mu.flatten(),posterior_cov,n_samps)
    predictions = []
    for w_s in w_samples:
        predictions.append(np.dot(w_s.T,X_val.T))

    return np.mean(predictions, axis=0)


# reduced = reduce_dimensions([X_train,X_test],X_train.shape[1], 3)
# X_train, X_test = np.real(reduced[0].T), np.real(reduced[1].T)

# X_train = np.delete(X_train, [2,5], axis=1)
# X_test = np.delete(X_test, [2,5], axis=1)

X_train, X_test = join_min_max(X_train, X_test)
X_temp, y_temp, X_val, Y_val = split_set(X_train, y_train, 40)

X_temp, y_temp = remove_outliers(X_temp, y_temp, 3)


# y_val = get_bayes_prediction(X_temp, y_temp, X_val)
y_val = score(X_val, Y_val, X_temp, y_temp)
print(y_val)

# Fit model and predict test values
y_pred = np.random.randint(y_train.min(), y_train.max(), X_test.shape[0])




# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission_bayes.csv', y_pred_pp, fmt='%d,%f,', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.

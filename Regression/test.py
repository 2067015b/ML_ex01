import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

X_train = np.loadtxt('reg_X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('reg_X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('reg_y_train.csv', delimiter=',', skiprows=1)[:, 1]


regr_1 = AdaBoostRegressor(RandomForestRegressor(max_depth=5))
regr_1.fit(X_train,y_train)



# Predict

y_1 = regr_1.predict(X_test)

# Plot the results
print(y_1)

test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_1
np.savetxt('my_submission_02.csv', y_pred_pp, fmt='%d,%f', delimiter=",",
           header=test_header, comments="")
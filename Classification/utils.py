import random

import numpy as np


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



def PCA(dataset, initial_dim, target_dim):
    cov = np.cov(dataset[0].T)

    eig_vals, eig_vec = np.linalg.eig(cov)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vec[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    matrix_w = eig_pairs[0][1].reshape(initial_dim, 1)

    for i in range(1, target_dim):
        matrix_w = np.hstack((matrix_w, eig_pairs[i][1].reshape(initial_dim, 1)))

    result = []
    for set in dataset:
        result.append(matrix_w.T.dot(set.T))

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

def compare(y1,y2):
    y1 = np.loadtxt(y1, delimiter=',', skiprows=1)[:, 1]
    y2 = np.loadtxt(y2, delimiter=',', skiprows=1)[:, 1]

    i = []
    total = 0
    for y1_, y2_ in zip(y1,y2):
        if y1_ == y2_:
            i.append(True)
        else:
            total+=1
            i.append(False)
    print("Total differences: {}".format(total))
    return i

def get_feature_stats(x, y, score, iters=300):
    for feature in range(x.shape[1]):
        sc = 0.0
        for _ in range(iters):
            X_t = np.delete(x, feature, axis=1)

            X_temp, y_temp, X_val, Y_val = split_set(X_t, y, 20)

            sc += score(X_temp, y_temp, X_val, Y_val)


def join_min_max(X_train, X_test):
    memory_train = np.reshape(np.absolute(np.subtract(X_train[:, 2],X_train[:, 1])),(X_train.shape[0], 1))

    cache_train = np.reshape(np.absolute(np.subtract(X_train[:, 5], X_train[:, 4])), (X_train.shape[0],1))

    memory_test = np.reshape(np.absolute(np.subtract(X_test[:, 2], X_test[:, 1])), (X_test.shape[0], 1))

    cache_test =np.reshape(np.absolute(np.subtract(X_test[:, 5], X_test[:, 4])), (X_test.shape[0], 1))

    X_train = np.delete(X_train, [1, 2, 4, 5], axis=1)
    X_test = np.delete(X_test, [1, 2, 4, 5], axis=1)

    # print("m_train: {}, m_test: {}, c_train: {}. c_test: {}. X_train: {}. X_test: {}".format(memory_train.shape, memory_test.shape, cache_train.shape, cache_test.shape, X_train.shape,X_test.shape))

    return np.hstack((X_train, memory_train, cache_train)), np.hstack((X_test, memory_test, cache_test))


def remove_outliers(X_train, y_train, std=3):
    prior_mean = np.mean(X_train, axis=0)
    prior_std = np.std(X_train, axis=0)

    delete = []
    for data_point in range(X_train.shape[0]):
        for feature in range(X_train.shape[1]):
            if abs(X_train[data_point, feature] - prior_mean[feature]) > prior_std[feature]*std:
                delete.append(data_point)

    return np.delete(X_train, delete, axis=0), np.delete(y_train, delete, axis=0)

if __name__ == "__main__":
    compare('my_submission_log_reg_orig.csv', 'my_submission_log_reg_best.csv')

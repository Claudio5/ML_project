import numpy as np
from implementations import *

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def split_nparts(array, n):
    k, m = divmod(len(array), n)
    return (array[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def get_split_indexes(x, y, k_fold, seed=1):
    """split the dataset based on the split ratio."""

    # Set seed
    np.random.seed(seed)

    # Generate random indices
    subdivision = int(len(x)/k_fold)
    indices = np.random.permutation(len(y))

    index_split_te = list(split_nparts(indices, k_fold))
    index_split_tr = np.zeros((k_fold, len(x) - subdivision))

    for i in range(0, k_fold):
        index_split_tr[i,:] = list(set(range(x.shape[0])) - set(index_split_te[i]))

    return index_split_te, index_split_tr

def cross_validation(optim_method, loss_function, tx, y, indexes_te, indexes_tr,
                    k_fold, args_optim = (), args_loss = ()):
    err_tr_list = []
    err_te_list = []
    accuracy_list = []
    for i in range(k_fold):
        x_te = x[indexes_te[i]]
        y_te = y[indexes_te[i]]
        x_tr = x[(indexes_tr[i]).astype(int)]
        y_tr = y[(indexes_tr[i]).astype(int)]

        x_tr, x_te = standardize(x_tr, True, x_te)

        w, err_tr = optim_method(y_tr, x_tr, *args_optim)

        err_te = loss_function(y, tx, w, *args_loss)

        y_predicted = predict_labels(w, x_te)

        accuracy_list.append(np.sum(np.equal(y_predicted, y_te)/len(y_te)))

        err_tr_list.append(err_tr)
        err_te_list.append(err_te)

    mse_tr_mean = np.mean(err_tr_list)
    mse_te_mean = np.mean(err_te_list)
    accuracy_mean = np.mean(accuracy_list)

    return mse_tr_mean, mse_te_mean, accuracy_mean


def standardize(x_tr, isTestingData = False, x_te = None):
    """ Standardize the testing data by substracting the mean and dividing
    by the variance. If isTestingData is true it standardize the testing data
    only using the training data """

    centered_data = x_tr - np.mean(x_tr, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)

    if(isTestingData and x_te is not None):
        centered_data_te = x_te - np.mean(x_tr, axis=0)
        std_data_te = centered_data_te / np.std(centered_data, axis=0)

    return std_data, std_data_te

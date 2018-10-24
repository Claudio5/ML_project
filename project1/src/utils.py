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
    # set seed
    np.random.seed(seed)
    # generate random indices
    subdivision = int(len(x)/k_fold)
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split_te = list(split_nparts(indices, k_fold))
    index_split_tr = np.zeros((k_fold, len(x) - subdivision))

    for i, ind_te in enumerate(index_split_te):
        index_split_tr[i,:] = [ind for ind in indices if ind not in list(index_split_te[i])]

    return index_split_te, index_split_tr

def cross_validation(optim_method, loss_func, tx, y, indexes_te, indexes_tr,
                    k_fold, *args_optim, *args_loss):
    for i in range(k_fold):
        x_te = x[indexes_te[i]]
        y_te = y[indexes_te[i]]
        x_tr = x[indexes_tr[i]]
        y_tr = y[indexes_tr[i]]

        x_tr = standardize(x_tr)

        w, err_tr = optim_method(y_tr, x_tr, *args_optim)

        err_te = loss_function(*args_loss)

        err_tr_list.append(err_tr)
        err_te_list.append(err_re)

    mse_tr_mean = np.mean(err_tr_list)
    mse_te_mean = np.mean(err_te_list)

    return mse_tr_mean, mse_te_mean


def standardize():
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)

    return std_data

x = np.array([[1,2],[1,3],[1,4],[1,9],[2,8],[1,90]])
y = np.array([1,4,3,8,6,6])
ratio = 0.75
random_seed = np.random.randint(0, 10000)

a,b = get_split_indexes(x,y,3)
print(a)

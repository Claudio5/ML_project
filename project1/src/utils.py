import numpy as np
from proj1_helpers import *
from implementations import *

wrong_value = -999

def build_poly(x, degree):
    """Polynomial basis functions for input data x"""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def split_nparts(array, n):
    """Split the array in n parts and returns every piece of the array"""
    k, m = divmod(len(array), n)
    return (array[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def get_split_indexes(x, y, k_fold):
    """Get the indexes of the dataset for training and for testing"""

    # Generate a random seed everytime we run the function
    np.random.seed(1)

    # Length of eaxh subdivision
    subdivision = int(len(x)/k_fold)

    # Permute all the indice to create the randomness
    indices = np.random.permutation(len(y))

    # Split the indices in k_fold parts, they respresent the testing indices
    index_split_te = list(split_nparts(indices, k_fold))
    index_split_tr = np.zeros((k_fold, len(x) - subdivision))

    # Finally get the training indices by substracting the testing k_indices
    # from the whole set
    for i in range(0, k_fold):
        index_split_tr[i,:] = list(set(range(x.shape[0])) - set(index_split_te[i]))

    return index_split_te, index_split_tr

def standardize(x, wrong_value, x_te = [], isTesting = False):
    """Standardize data by not taking into account in the statistics
    the 0's in the data"""
    mean_tr = x.copy()
    std_data_tr = x.copy()
    mean_tr_list = []
    std_tr_list = []

    # Compute the mean of the feature by not taking into account the 0
    # Thus since we are centering the data around 0 the values at 0 reamin unchanged
    for ind, val in enumerate(x.T):
        x_cleaned = val[val != wrong_value]
        if(len(x_cleaned > 0)):
            mean_tr[:,ind] = x[:,ind] - np.mean(x_cleaned)*(val != wrong_value)
            mean_tr_list.append(np.mean(x_cleaned))
        else:
            mean_tr_list.append(0)

    # Divide by the std again by not considering the 0
    for ind, val in enumerate(mean_tr.T):
        std_data_tr[:,ind] = mean_tr[:,ind] / np.std(val[x[:,ind] != wrong_value])
        std_tr_list.append(np.std(val[x[:,ind] != wrong_value]))

    std_data_te = []
    if(isTesting and x_te is not None):
        centered_data_te = x_te - mean_tr_list*(x_te != wrong_value)
        std_data_te = centered_data_te / std_tr_list
        std_data_te[std_data_te > 1000] = 1000

    return std_data_tr, std_data_te

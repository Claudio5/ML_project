import numpy as np
from proj1_helpers import *
from implementations import *

wrong_value = -999
MAX_SEED = 100000

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
    seed = np.random.randint(0,MAX_SEED)
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

def cross_validation(optim_method, loss_function, tx, y, indexes_te, indexes_tr,
                    k_fold, isBuildPoly = False, args_optim = (), args_loss = ()):
    """Cross validation of the training set for any optimization method and for
    any value of k_fold"""
    err_tr_list = []
    err_te_list = []
    accuracy_list = []
    for i in range(k_fold):
        x_te = tx[indexes_te[i]]
        y_te = y[indexes_te[i]]
        x_tr = tx[(indexes_tr[i]).astype(int)]
        y_tr = y[(indexes_tr[i]).astype(int)]

        if not isBuildPoly:
            #x_tr, x_te = standardize(x_tr, True, x_te)
            x_tr, x_te = standardize3(x_tr, wrong_value, x_te, True)
        else:
            # Does not take into account the column containing only ones to avoid a std of 0
            # It happens when we try to add polynomial features
            x_tr[:,1:], x_te[:,1:] = standardize3(x_tr[:,1:], wrong_value, x_te[:,1:], True)

        # x_tr[:,1:], mean_tr, std_tr = standardize2(x_tr[:,1:])
        # #x_te, mean, std = standardize(x_te)
        # x_te[:,1:] = standardize_given2(x_te[:,1:], mean_tr, std_tr)

        # Get the final value of w
        w, err_tr = optim_method(y_tr, x_tr, *args_optim)

        # Loss function corresponding to the method
        err_te = loss_function(y_te, x_te, w, *args_loss)

        y_predicted = predict_labels(w, x_te)

        # When doing logistic regression put again the testing preditions to -1
        y_te[y_te == 0] = -1

        # Compute the accuracy by checking how many values are corrected with
        # testing labels
        accuracy_list.append(np.sum(np.equal(y_predicted, y_te)/len(y_te)))

        err_tr_list.append(err_tr)
        err_te_list.append(err_te)

    # Compute the final statistics
    mse_tr_mean = np.mean(err_tr_list)
    mse_te_mean = np.mean(err_te_list)
    rmse_tr_mean = np.sqrt(2*mse_tr_mean)
    rmse_te_mean = np.sqrt(2*mse_te_mean)
    accuracy_mean = np.mean(accuracy_list)

    return mse_tr_mean, mse_te_mean, rmse_tr_mean, rmse_te_mean, accuracy_mean

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

def standardize3(x, wrong_value, x_te = [], isTesting = False):
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

    if(isTesting and x_te is not None):
        centered_data_te = x_te - mean_tr_list*(x_te != wrong_value)
        std_data_te = centered_data_te / std_tr_list
        std_data_te[std_data_te > 1000] = 1000

    return std_data_tr, std_data_te

def standardize4(x, wrong_value, x_te = [], isTesting = False):
    mean_tr = x.copy()
    std_data_tr = mean_tr.copy()
    mean_tr_list = []
    std_tr_list = []

    for ind, val in enumerate(x.T):
        x_cleaned = val[val != wrong_value]
        if(len(x_cleaned > 0)):
            mean_tr[:,ind] = x[:,ind] - np.mean(x_cleaned)*(val != wrong_value)
            mean_tr_list.append(np.mean(x_cleaned))
        else:
            mean_tr_list.append(0)

    mean_tr[mean_tr==wrong_value] = 0
    for ind, val in enumerate(mean_tr.T):
        std_data_tr[:,ind] = mean_tr[:,ind] / np.std(val[x[:,ind] != wrong_value])
        std_tr_list.append(np.std(val[x[:,ind] != wrong_value]))

    std_data_te = x_te
    if(isTesting and x_te is not None):
        centered_data_te = x_te - mean_tr_list*(x_te != wrong_value)
        centered_data_te[centered_data_te==wrong_value] = 0
        std_data_te = centered_data_te / std_tr_list
        std_data_te[std_data_te > 1000] = 1000

    return std_data_tr, std_data_te

def standardize2(x):
    """Standardize the original data set."""
    #standardize is done feature by feature to have equal weights.
    #values that are worth 0 will stay at 0. corrupted data (=-999) are set to 0 earlier
    mean_x = np.mean(x,axis=0)
    x = x - np.mean(x,axis=0)*(x!=0)
    std_x = np.std(x,axis=0)
    std_x_t = np.std(x,axis=0)*(x!=0)
    std_x_t[std_x_t==0] = 1
    x = x / std_x_t
    return x, mean_x, std_x

def standardize_given2(x, mean_x, std_x):
    """Standardize the original data set with given mean_x and std_x."""
    x = x - mean_x*(x!=0)
    std_x_t = std_x*(x!=0)
    std_x_t[std_x_t==0] = 1
    x = x / std_x_t
    x[(x>1000)] = 1000  #handle outliers
    return x

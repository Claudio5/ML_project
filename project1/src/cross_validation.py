import numpy as np
from proj1_helpers import *
from implementations import *
from utils import *

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
            x_tr, x_te = standardize(x_tr, 0, x_te, True)
        else:
            # Does not take into account the column containing only ones to avoid a std of 0
            # It happens when we try to add polynomial features
            x_tr[:,1:], x_te[:,1:] = standardize(x_tr[:,1:], 0, x_te[:,1:], True)


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

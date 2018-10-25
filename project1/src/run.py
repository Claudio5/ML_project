#importing libraries
import numpy as np

from proj1_helpers import *
from implementations import *
from utils import *

def adjust_wrong_data(training_data, y, wrongValue, testing_data = []):
    #Assign the mean value of the uncorrupted data of a feature to the corrupted data of this feature.
    #If training data, then the corrupted data take the mean value of the uncorrupted training data of the same class.
    #If testing data, then the corrupted data take the mean value of all the uncorrpted training data of this feature.

    if len(testing_data) == 0:
        for i in range(np.shape(training_data)[1]):
            feature = training_data[:,i]
            feature_class_1 = feature[y==-1]
            feature_class_1_clean = feature_class_1[feature_class_1!=wrongValue]
            feature_mean_1 = np.mean(feature_class_1_clean)
            feature_class_2 = feature[y==1]
            feature_mean_2 = np.mean(feature_class_2[feature_class_2!=wrongValue])
            for j in range(np.shape(training_data)[0]):
                if y[j] == -1 and training_data[j,i] == wrongValue:
                    training_data[j,i] = feature_mean_1
                if y[j] == 1 and training_data[j,i] == wrongValue:
                    training_data[j,i] = feature_mean_2
        return training_data

    else:
        for i in range(np.shape(training_data)[1]):
            feature = training_data[:,i]
            feature_clean = feature[feature!=wrongValue]
            feature_mean = np.mean(feature_clean)
            for j in range(np.shape(testing_data)[0]):
                if testing_data[j,i] == wrongValue:
                    testing_data[j,i] = feature_mean
        return testing_data

training_pred, training_data, ids_tr = load_csv_data("../data/train.csv")
testing_pred, testing_data, ids_te = load_csv_data("../data/test.csv")

wrong_value = -999
training_data = adjust_wrong_data(training_data, training_pred, wrong_value)
testing_data = adjust_wrong_data(training_data, training_pred, wrong_value, testing_data)

poly_degree = 2
x_tr = build_poly(training_data, poly_degree)
x_te = build_poly(testing_data, poly_degree)
initial_w = np.zeros(x_tr.shape[1])
lambda_ = 1e-9

x_tr[:,1:], x_te[:,1:] = standardize(x_tr[:,1:], True, x_te[:,1:])

w, err = ridge_regression(training_pred, x_tr, lambda_)

y_predicted = predict_labels(w, x_te)

create_csv_submission(ids_te, y_predicted, 'pred.csv')

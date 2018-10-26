#importing libraries
import numpy as np

from proj1_helpers import *
from implementations import *
from utils import *

training_pred, training_data, ids_tr = load_csv_data("../data/train.csv")
testing_pred, testing_data, ids_te = load_csv_data("../data/test.csv")

training_data[training_data==-999] = 0
testing_data[testing_data==-999] = 0
training_data[:,22] += 1
testing_data[:,22] += 1

poly_degree = 12
x_tr = build_poly(training_data, poly_degree)
#x_tr = add_features_cross(x_tr, training_data)
x_te = build_poly(testing_data, poly_degree)
#x_te = add_features_cross(x_te, testing_data)
initial_w = np.zeros(x_tr.shape[1])
lambda_ = 1e-15

x_tr[:,1:], x_te[:,1:] = standardize3(x_tr[:,1:], 0, x_te[:,1:], True)

w, err = ridge_regression(training_pred, x_tr, lambda_)

y_predicted = predict_labels(w, x_te)

create_csv_submission(ids_te, y_predicted, 'pred.csv')

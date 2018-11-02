#importing libraries
import numpy as np

from proj1_helpers import *
from implementations import *
from utils import *

wrong_value = -999
epsilon = 0.0000001

training_pred, training_data, ids_tr = load_csv_data("../data/train.csv")
testing_pred, testing_data, ids_te = load_csv_data("../data/test.csv")

# Put the erronous data to 0
training_data[training_data==wrong_value] = 0
testing_data[testing_data==wrong_value] = 0

# Categoritical data, we add 1 to avoid 0 label
training_data[:,22] += 1
testing_data[:,22] += 1

# For the values to be different from 0 before the standardization
training_data[:,12] += epsilon
testing_data[:,12] += epsilon

# Deleting features
training_data = np.delete(training_data, 29, axis = 1)
training_data = np.delete(training_data, 28, axis = 1)
testing_data = np.delete(testing_data, 29, axis = 1)
testing_data = np.delete(testing_data, 28, axis = 1)

poly_degree = 16
lambda_ = 1e-15
x_tr = build_poly(training_data, poly_degree)
x_te = build_poly(testing_data, poly_degree)

x_tr[:,1:], x_te[:,1:] = standardize(x_tr[:,1:], 0, x_te[:,1:], True)

w, err = ridge_regression(training_pred, x_tr, lambda_)
y_predicted = predict_labels(w, x_te)
create_csv_submission(ids_te, y_predicted, 'pred.csv')

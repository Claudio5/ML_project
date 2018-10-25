import numpy as np

def compute_mse(y, tx, w):
    e = y - tx.dot(w)
    return 1/(2*len(y)) * np.dot(e, e)

def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(y) / len(y)
    return grad, err

def compute_SGD_gradient(y,tx,w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(y)
    return grad, err

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Least Squares gradient descent algorithm"""
    w = initial_w
    for n in range(max_iters):
        # compute gradient
        grad, err = compute_gradient(y, tx, w)

        # gradient w by descent update
        w = w - gamma * grad

    loss = compute_mse(y, tx, w)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    w = initial_w

    for n_iter in range(max_iters):
        index = np.random.randint(0, len(y) - 1)
        # Compute a stochastic gradient and loss
        grad, err = compute_SGD_gradient(y[index], tx[index, :], w)

        # Update w through the stochastic gradient update
        w = w - gamma * grad

    loss = compute_mse(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def sigmoid(x):
    return 1/(1+np.exp(-x))

def log_likelihood_gradient(y, tx, w):
    return np.dot(np.transpose(tx), sigmoid(np.dot(tx, w)) - y)/len(y)

def loss_logistic_regression(y, tx, w):
    return np.sum(np.log(1 + np.exp(np.dot(tx, w))) - y*np.dot(tx, w))/len(y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad = log_likelihood_gradient(y, tx, w)
        w = w - gamma*grad

    loss = loss_logistic_regression(y, tx, w)
    return w, loss

def reg_logistic_regression_loss(y, tx, w, lambda_):
    return (np.sum(np.log(1 + np.exp(np.dot(tx, w)))) - y.dot(np.dot(tx, w)))/len(y) + lambda_/2*np.vdot(w, w)

def log_likelihood_regularized_gradient(y, tx, w, lambda_):
    return np.dot(np.transpose(tx), sigmoid(np.dot(tx, w)) - y)/len(y) + lambda_*w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for n in range(max_iters):
        grad = log_likelihood_regularized_gradient(y, tx, w, lambda_)
        w = w - gamma*grad
    loss = reg_logistic_regression_loss(y, tx, w, lambda_)
    return w, loss

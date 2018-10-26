import numpy as np

def compute_mse(y, tx, w):
    """Computes the mean squared error betweenw"""
    e = y - tx.dot(w)
    return 1/(2*len(y)) * np.dot(e, e)

def compute_mse_gradient(y, tx, w):
    """Computes the gradient of the mean squared error with
    respect to w"""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(y)
    return grad, err

def compute_mse_SGD_gradient(y,tx,w):
    """Computes the gradient of the loss for stochastic gradient descent"""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err)
    return grad, err

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent for linear regression"""
    w = initial_w
    for n in range(max_iters):
        # Compute gradient
        grad, err = compute_mse_gradient(y, tx, w)

        # Gradient w by descent update
        w = w - gamma * grad

    loss = compute_mse(y, tx, w)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent for linear regression"""
    w = initial_w
    for n_iter in range(max_iters):
        # Define the index of the sample that we will pick
        index = np.random.randint(0, len(y) - 1)

        # Compute a stochastic gradient and loss for the picked sample
        grad, err = compute_mse_SGD_gradient(y[index], tx[index, :], w)

        # Update w through the stochastic gradient update
        w = w - gamma * grad

    loss = compute_mse(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution using normal equations"""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    # The soluion to the system will give us the minimized w
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def sigmoid(x):
    """Sigmoid function"""
    return 1/(1+np.exp(-x))

def logistic_loss_gradient(y, tx, w):
    """Gradient of the log likelihood with respect to the weight vector"""
    return np.dot(np.transpose(tx), sigmoid(np.dot(tx, w)) - y)/len(y)

def loss_logistic_regression(y, tx, w):
    """Loss function for logistic regression"""
    return np.sum(np.log(1 + np.exp(np.dot(tx, w))) - y*np.dot(tx, w))/len(y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        # Compute the gradient of the loss function for logistic regression
        grad = logistic_loss_gradient(y, tx, w)

        # Update the weight vector by the computed gradient
        w = w - gamma * grad

    loss = loss_logistic_regression(y, tx, w)
    return w, loss

def reg_logistic_regression_loss(y, tx, w, lambda_):
    """Loss function for regularized logistic regression"""
    return (np.sum(np.log(1 + np.exp(np.dot(tx, w)))) -
            y.dot(np.dot(tx, w)))/len(y) + lambda_/2*np.vdot(w, w)

def reg_logistic_loss_gradient(y, tx, w, lambda_):
    """Gradient of the loss for the regularized logistic regression"""
    return np.dot(np.transpose(tx), sigmoid(np.dot(tx, w)) - y)/len(y) + lambda_*w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent"""
    w = initial_w
    for n in range(max_iters):
        # Compute the gradient of the loss function for logistic regression
        grad = reg_logistic_loss_gradient(y, tx, w, lambda_)

        # Update the weight vector by the computed gradient
        w = w - gamma*grad

    loss = reg_logistic_regression_loss(y, tx, w, lambda_)
    return w, loss

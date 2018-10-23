import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *


def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.

    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model

    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    # L = compute_cost(A3, Y)
    L = -np.sum(np.multiply(Y, np.log(A3)) + np.multiply((1-Y), np.log(1-A3))) / m

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    R = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2*m)

    return L + R


def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """



def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, plot_cost=True, lambd=0, keep_prob=1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.

    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """

    m = X.shape[1]
    layer_dims = [X.shape[0], 20, 3, 1]
    parameters = initialize_parameters(layer_dims)

    costs = []

    for i in range(num_iterations):
        a3, cache = forward_propagation(X, parameters)

        cost = compute_cost_with_regularization(a3, Y, lambd)

        grads = backward_propagation(X, Y, cache)

        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration {}: {}".format(i, cost))

    if plot_cost:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'


    # train_X, train_Y, test_X, test_Y = load_2D_dataset(False)
    #
    # # Train the model without any regularization
    # parameters = model(train_X, train_Y)
    # print("On the training set:")
    # predictions_train = predict(train_X, train_Y, parameters)
    # print("On the test set:")
    # predictions_test = predict(test_X, test_Y, parameters)
    #
    # plt.title("Model without regularization")
    # axes = plt.gca()
    # axes.set_xlim([-0.75, 0.40])
    # axes.set_ylim([-0.75, 0.65])
    # plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
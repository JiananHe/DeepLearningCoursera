import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from dnn_functions import initialize_parameters_deep, L_model_backward, L_model_forward, update_parameters, compute_cost


# GRADED FUNCTION: n_layer_model
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, plot_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(1)
    parameters = initialize_parameters_deep(layers_dims)

    costs = []  # keep track of cost
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    # plot the cost
    if plot_cost:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """
    m = X.shape[1]
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print(np.sum((p == y)))
    print("Accuracy: " + str(float(np.sum((p == y))) / m))

    return p


def print_mislabeled_images(classes, X, y, p):
    """
    :param classes: example classes(cat or non-cat)
    :param X: data set of examples
    :param y: labels of examples
    :param p: predicts of examples
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])

    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0, index])].decode("utf-8") +
                  " \n Class: " + classes[y[0, index]].decode("utf-8"))

    plt.show()


if __name__=="__main__":
    # Load and standard dataset
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

    train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T / 255
    test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T / 255

    # Training the model
    layers_dims = [12288, 20, 7, 5, 1]  # 5-layer model
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True, plot_cost=False)

    # Model accuracy
    p_train = predict(train_x, train_y, parameters)
    p_test = predict(test_x, test_y, parameters)

    # Plot the mislabeled images
    # print_mislabeled_images(classes, test_x, test_y, p_test)

    # Try own images
    my_image = "my_image.JPG"
    image = np.array(ndimage.imread(my_image, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64, 64)).reshape(64 * 64 * 3, 1)
    my_label = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
    my_predict = predict(my_image, my_label, parameters)

    plt.imshow(image)
    plt.title("Prediction: " + classes[int(my_predict[0])].decode("utf-8") +
              " \n Class: " + classes[int(my_label[0])].decode("utf-8"))
    plt.show()

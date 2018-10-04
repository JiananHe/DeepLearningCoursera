import matplotlib.pyplot as plt
import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def preprocess_dataset(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes):
    """
    Preporocess dataset
    
    Arguments:
    train_set_x_orig -- training set represented by a numpy array of shape (m_train, num_px,  num_px, channels)
	train_set_y -- training labels represented by a numpy array (vector) of shape (1, m_train)
    test_set_x_orig -- test set represented by a numpy array of shape (m_test, num_px,  num_px, channels)
	test_set_y -- test labels represented by a numpy array (vector) of shape (1, m_test)

    Returns:
    train_set_x -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
	test_set_x -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    """

    # Example of a picture
    # index = 26
    # plt.imshow(train_set_x_orig[index])
    # # Turn off the interactive method
    # plt.ioff()
    # plt.show()
    # print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
    #     "utf-8") + "' picture.")

    ### START CODE HERE ### (≈ 3 lines of code)
    m_train = train_set_y.shape[1]
    m_test = test_set_y.shape[1]
    num_px = train_set_x_orig.shape[1]
    num_py = train_set_x_orig.shape[2]
    num_pchannels = train_set_x_orig.shape[3]
    ### END CODE HERE ###

    print("Number of training examples: m_train = " + str(m_train))
    print("Number of testing examples: m_test = " + str(m_test))
    print("Height/Width of each image: num_px = " + str(num_px))
    print("Number of channels of each image: num_pchannels = " + str(num_pchannels))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_py) + ", " + str(num_pchannels) + ")")
    print("train_set_x shape: " + str(train_set_x_orig.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x shape: " + str(test_set_x_orig.shape))
    print("test_set_y shape: " + str(test_set_y.shape))

    # Reshape the training and test examples

    ### START CODE HERE ### (≈ 2 lines of code)
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    ### END CODE HERE ###

    print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print("test_set_y shape: " + str(test_set_y.shape))
    print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

    # Standardize dataset
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255
    return train_set_x, test_set_x

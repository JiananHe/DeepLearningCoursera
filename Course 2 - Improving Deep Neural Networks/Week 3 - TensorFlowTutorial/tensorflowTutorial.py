import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time

np.random.seed(1)


def tutorials():
    y_hat = tf.constant(36, name='y_hat')  # Define y_hat constant. Set to 36.
    y = tf.constant(39, name='y')

    loss = tf.Variable((y-y_hat)**2, name='loss')  # Create a variable for the loss

    init = tf.global_variables_initializer()  # When init is run later (session.run(init)),
                                              # the loss variable will be initialized and ready to be computed
    with tf.Session() as session:             # Create a session and print the output
        session.run(init)                     # Initializes the variables
        print(session.run(loss))              # Prints the loss

    a = tf.constant(2)
    b = tf.constant(10)
    c = tf.multiply(a, b)                    # put in the 'computation graph', does not compute c yet
    print(c)                                # Tensor("Mul:0", shape=(), dtype=int32)

    sess = tf.Session()
    print(sess.run(c))                      # 20

    # placeholder
    x = tf.placeholder(tf.int64, name='x')
    print(sess.run(x**2, feed_dict={x: 3}))
    sess.close()


def linear_function():
    """
    Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """

    np.random.seed(1)

    W = np.random.rand(4, 3)
    X = np.random.rand(3, 1)
    b = np.random.rand(4, 1)

    Y = tf.add(tf.matmul(W, X), b)

    sess = tf.Session()
    result = sess.run(Y)
    sess.close()

    return result


def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- input value, scalar or vector

    Returns:
    results -- the sigmoid of z
    """
    x = tf.placeholder(tf.float32, name='x')
    sigmoid = tf.sigmoid(x)

    with tf.Session() as session:
        result = session.run(sigmoid, feed_dict={x: z})

    return result


def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy

    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0)

    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels"
    in the TensorFlow documentation. So logits will feed into z, and labels into y.

    Returns:
    cost -- runs the session of the cost (formula (2))
    """

    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    with tf.Session() as session:
        cost = session.run(cost, feed_dict={z: logits, y: labels})

    return cost


def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot


def ones(shape):
    """
    Creates an array of ones of dimension shape

    Arguments:
    shape -- shape of the array you want to create

    Returns:
    ones -- array containing only ones
    """
    ones = tf.ones(shape)

    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()

    return ones


if __name__ == "__main__":
    # tutorials()
    # print("result = " + str(linear_function()))
    # print("sigmoid(0) = " + str(sigmoid(0)))
    # print("sigmoid(12) = " + str(sigmoid(12)))

    # logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
    # cost = cost(logits, np.array([0, 0, 1, 1]))
    # print("cost = " + str(cost))

    # labels = np.array([1, 2, 3, 0, 2, 1])
    # one_hot = one_hot_matrix(labels, C=4)
    # print("one_hot = " + str(one_hot))

    print("ones = " + str(ones([3])))

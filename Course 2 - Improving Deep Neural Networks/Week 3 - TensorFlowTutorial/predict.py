import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time
import scipy
from PIL import Image
from scipy import ndimage
from numberGestureRecognition import *

# We preprocess your image to fit your algorithm.
fname = ".\\my_images\\gestureTwo.jpg"
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T

# read variable
parameters, _ = initialize_parameters()
saver = tf.train.Saver()

with tf.Session() as sess:
    # load model
    saver.restore(sess, ".\\model\\gesture_model")
    parameters = sess.run(parameters)
    my_image_prediction = tf_utils.predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
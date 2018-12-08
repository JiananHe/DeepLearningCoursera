import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
from keras.models import load_model
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


if __name__ == "__main__":
    # load trained model
    happyModel = load_model('model.h5')

    # test/evaluate the model.
    # X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # X_test = X_test_orig / 255.
    # Y_test = Y_test_orig.T
    #
    # preds = happyModel.evaluate(x=X_test, y=Y_test, batch_size=32, verbose=1, sample_weight=None)
    # print("Loss = " + str(preds[0]))
    # print("Test Accuracy = " + str(preds[1]))

    # prints the details of layers in a table with the sizes of its inputs/outputs
    happyModel.summary()

    # plots your graph in a nice layout.
    plot_model(happyModel, to_file='happModel.png')
    SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))

    # predict a picture
    img = image.load_img('images\\unhappy.jpg', target_size=(64, 64))
    plt.imshow(img)
    plt.show()

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print(happyModel.predict(x))
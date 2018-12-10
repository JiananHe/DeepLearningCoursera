import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model = load_model('model.h5')

    # prints the details of layers in a table with the sizes of its inputs/outputs
    # model.summary()

    # plots your graph in a nice layout.
    # plot_model(model, to_file='ResNet50.png')
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))

    # predict my own picture
    rootdir = '.\\images'
    img_list = os.listdir(rootdir)
    img_num = len(img_list)
    print(img_list)
    for i in range(img_num):
        img_path = os.path.join(rootdir, img_list[i])
        img = image.load_img(img_path, target_size=(64, 64))

        x = image.img_to_array(img) / 255.
        x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        pred = model.predict(x)
        print(pred)

        plt.subplot(2, img_num/2, i+1)
        plt.imshow(img)
        plt.title(np.argmax(np.squeeze(pred)))

    plt.show()



    # predict picture in test
    # X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # X_test = X_test_orig / 255.
    # Y_test = convert_to_one_hot(Y_test_orig, 6).T
    #
    # preds = model.evaluate(X_test, Y_test)
    # print("Loss = " + str(preds[0]))
    # print("Test Accuracy = " + str(preds[1]))
    #
    # print(X_test_orig.shape)
    # n = X_test_orig.shape[0]
    # rs = np.random.choice(n, 10, replace=False)
    # print(rs)
    # plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    # for i in range(len(rs)):
    #     index = rs[i]
    #     img = X_test[index]
    #     print(Y_test[index])
    #
    #     # x = image.img_to_array(img)
    #     # x = np.expand_dims(x, axis=0)
    #     # x = preprocess_input(x)
    #     # print(x.shape)
    #
    #     x = np.expand_dims(img, axis=0)
    #     pred = model.predict(x)
    #     print(pred)
    #
    #     plt.subplot(2, 5, i+1)
    #     plt.imshow(img, interpolation='nearest')
    #     plt.title(np.argmax(np.squeeze(pred)))
    #
    # plt.show()
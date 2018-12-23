from faceModel import *
from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import pickle as pkl


def dao():
    """
    read pkl file to get faces encodings database

    :return: database: faces encodings database with dict type
    """
    with open('face_encodings.pkl', 'rb') as f:
        database = pkl.load(f)

    return database


def verify(image_path, identity, database, model):
    """
        Function that verifies if the person on the "image_path" image is "identity".

        Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras

        Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
        """
    encoding = img_to_encoding(image_path, model)

    # Compute distance with identity's image
    dist = np.linalg.norm(encoding-database[identity])

    # Open the door if dist < 0.7, else don't open
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open


def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    encoding = img_to_encoding(image_path, model)

    min_dist = 100
    identity = ""

    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding-db_enc)
        print("the distance with " + name + " is: " + str(dist))
        if dist < min_dist:
            identity = name
            min_dist = dist

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity


if __name__=="__main__":
    # load trained model
    FRmodel = load_model('FRmodel.h5', custom_objects={'triplet_loss': triplet_loss})

    database = dao()

    verify("images/andrew.jpg", "felix", database, FRmodel)

    who_is_it("images/zixia_2.jpg", database, FRmodel)


from keras.models import load_model
from keras import backend as K
K.set_image_data_format('channels_first')
from fr_utils import *
from inception_blocks_v2 import *
import pickle as pkl

np.set_printoptions(threshold=np.nan)


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negtive = y_pred[0], y_pred[1], y_pred[2]

    # Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(positive, anchor)))
    # Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(negtive, anchor)))
    # subtract the two previous distances and add alpha.
    basic_loss = tf.add(alpha, tf.subtract(pos_dist, neg_dist))
    # subtract the two previous distances and add alpha.
    loss = tf.maximum(0.0, basic_loss)

    return loss


def face_encoding_db(FRmodel):
    database = {}
    database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
    database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
    database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
    database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
    database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
    database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
    database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
    database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
    database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
    database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
    database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
    database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
    database["zixia"] = img_to_encoding("images/zixia_1.jpg", FRmodel)

    # save to a pkl file
    file = open('face_encodings.pkl', 'wb+')
    pkl.dump(database, file)
    file.close()


if __name__=="__main__":
    # with tf.Session() as test:
    #     tf.set_random_seed(1)
    #     y_true = (None, None, None)
    #     y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
    #               tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
    #               tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
    #     loss = triplet_loss(y_true, y_pred)
    #
    #     print("loss = " + str(loss.eval()))

    # FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    # print("Total Params:", FRmodel.count_params())
    # FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    # # Loading the trained model
    # load_weights_from_FaceNet(FRmodel)
    # FRmodel.save('FRmodel.h5')

    FRmodel = load_model('FRmodel.h5', custom_objects={'triplet_loss': triplet_loss})

    face_encoding_db(FRmodel)

import lr_utils
import training
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
num_px = train_set_x_orig.shape[1]
num_py = train_set_x_orig.shape[2]

# Preprocess the data (reshape and standardize dataset)
train_set_x, test_set_x = lr_utils.preprocess_dataset(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes)
print()

# Train the model
d = training.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
                   print_cost=True)
print()

# Example of a picture that was wrongly classified.(overfitting the training data)
# index = 5
# plt.imshow(test_set_x[:, index].reshape((num_px, num_py, train_set_x_orig.shape[3])))
# plt.show()
# print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" +
#       classes[int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture.")


# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# Choice of learning rate
# learning_rates = [0.01, 0.005, 0.0002, 0.001, 0.0005, 0.0001]
# models = {}
# for i in learning_rates:
#     print ("learning rate is: " + str(i))
#     models[str(i)] = training.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
#     print ('\n' + "-------------------------------------------------------" + '\n')
#
# for i in learning_rates:
#     plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
#
# plt.ylabel('cost')
# plt.xlabel('iterations')
#
# legend = plt.legend(loc='upper center', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()


# Test with your own image
my_image = "my_image.jpg"
image = np.array(ndimage.imread(my_image, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape(num_px * num_px * 3, 1)
my_predicted_image = training.predict(d["w"], d["b"], my_image)

plt.imshow(image)
plt.imshow(my_image.reshape(num_px, num_px, 3))
plt.show()
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
      classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
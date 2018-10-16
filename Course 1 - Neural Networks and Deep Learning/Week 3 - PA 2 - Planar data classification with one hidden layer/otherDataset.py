import matplotlib.pyplot as plt
import nnModel
from testCases import *
from planar_utils import plot_decision_boundary, load_extra_datasets


# Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

dataset = "blobs"
X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y % 2

print('The shape of X is: ' + str(X.shape))
print('The shape of Y is: ' + str(Y.shape))
print('I have m = %d training examples!' % (Y.shape[1]))

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
plt.show()

# Training the nn model
parameters = nnModel.nn_model(X, Y, 20)

# Plot the decision boundary
plot_decision_boundary(lambda x: nnModel.predict(parameters, x.T), X, np.squeeze(Y))
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

# Print accuracy
predictions = nnModel.predict(parameters, X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

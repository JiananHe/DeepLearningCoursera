import matplotlib.pyplot as plt
from testCases import *
import sklearn.linear_model
from planar_utils import plot_decision_boundary, load_planar_dataset

np.random.seed(1)  # set a seed so that the results are consistent


X, Y = load_planar_dataset()

print('The shape of X is: ' + str(X.shape))
print('The shape of Y is: ' + str(Y.shape))
print('I have m = %d training examples!' % (Y.shape[1]))

Y_squeeze = np.squeeze(Y)

# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y_squeeze, s=40, cmap=plt.cm.Spectral)
plt.show()

# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y_squeeze.T)

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y_squeeze)
plt.title("Logistic Regression")
plt.show()

# Print accuracy
LR_prediction = clf.predict(X.T)
print('Accuracy of logistic regression: %f'
      % float((np.dot(Y, LR_prediction) + np.dot(1-Y, 1-LR_prediction)) / float(Y.size) * 100) +
      '%' + "(percentage of correctly labelled datapoints)")
# print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from  sklearn import  datasets
from sklearn import metrics
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

# Load data
iris = load_iris()
from sklearn.model_selection import train_test_split


iris=datasets.load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    # print("PairIDX",pair)
    X = x_train[:, pair]
    y = y_train
    X1 = x_test[:, pair]
    y1 = y_test

    # Train
    # clf = DecisionTreeClassifier(max_depth=10).fit(X, y)
    # clf = Perceptron(max_iter=40, eta0=0.1, random_state=1).fit(X,y)
    clf = MLPClassifier(hidden_layer_sizes=(4), solver='sgd', learning_rate_init=0.01, max_iter=500,alpha=10).fit(X,y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
    y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y1 == i)
        plt.scatter(X1[idx, 0], X1[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of MLP using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
#
# plt.figure()
# # clf = DecisionTreeClassifier().fit(iris.data, iris.target)
# # plot_tree(clf, filled=True)
plt.show()

#print(confusion_matrix(y_test,Z))

from sklearn import tree
# classifier=tree.DecisionTreeClassifier(max_depth=10)
classifier = MLPClassifier(hidden_layer_sizes=(4), solver='sgd', learning_rate_init=0.01, max_iter=500,alpha=10)
# classifier = Perceptron(max_iter=100, eta0=0.01, random_state=1)
classifier.fit(x_train,y_train)
predictions=classifier.predict(x_test)
print(classification_report(y1, predictions))
from sklearn.metrics import accuracy_score
print("Accuracy", accuracy_score(y_test,predictions))
result = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(result)
# disp = metrics.plot_confusion_matrix(classifier, x_test, y_test,
#                                  display_labels=cn,
#                                  cmap=plt.cm.Blues,
#                                  normalize=None)
# disp.ax_.set_title('Decision Tree Confusion matrix, without normalization');
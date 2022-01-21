from sklearn import datasets
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler

rcParams["figure.figsize"]=10,5

iris = datasets.load_iris()
x = iris.data[:, :]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1, stratify = y)
print ('Lables count in y:', np.bincount(y))
print('Lables counts in y_train:', np.bincount(y_train))
print('Lables counts in y_test:', np.bincount(y_test))

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

x_train_std = x_train_std[:,[2,3]]
x_test_std = x_test_std[:,[2,3]]

ppn = Perceptron(max_iter = 40, eta0=0.1, random_state=1)
ppn.fit(x_train_std, y_train)
y_pred = ppn.predict(x_test_std)
print('Misclassified samples: %d' %(y_test !=y_pred).sum())
from sklearn.metrics import accuracy_score
print("Accuracy:%.2f" %accuracy_score(y_test, y_pred))
print("Accuracy:%.2f" %ppn.score(x_test_std, y_test))


for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Train
    clf = DecisionTreeClassifier().fit(X, y)
    clf = Perceptron(max_iter=40, eta0=0.1, random_state=1)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
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
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
#
plt.figure()
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plt.show()

from  sklearn import  datasets
from sklearn import metrics
iris=datasets.load_iris()
x=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)
from sklearn import tree
classifier=tree.DecisionTreeClassifier()
classifier.fit(x_train,y_train)
predictions=classifier.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))


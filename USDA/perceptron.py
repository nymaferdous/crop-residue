import numpy as np
# manipulating data via DataFrames, 2-D tabular, column-oriented data structure
import pandas as pd
# producing plots and other 2D data visualizations. Use plotly if you want interactive graphs
import matplotlib.pyplot as plt
# statistical visualizations (a wrapper around Matplotlib)
import seaborn as sns
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
from sklearn.metrics import classification_report,confusion_matrix
warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)

# Assign the csv data to a DataFrame
data = pd.read_csv("./Iris.csv")
data.head(10).style

# # Plot a histogram of SepalLength Frequency on Species (matplotlib)
# Iris_setosa = data[data["Species"] == "Iris-setosa"]
# Iris_versicolor = data[data["Species"] == "Iris-versicolor"]
# Iris_virginica = data[data["Species"] == "Iris-virginica"]
#
# Iris_setosa["SepalLengthCm"].plot.hist(alpha=0.5,color='blue',bins=50) # Setting the opacity(alpha value) & setting the bar width((bins value)
# Iris_versicolor["SepalLengthCm"].plot.hist(alpha=0.5,color='green',bins=50)
# Iris_virginica["SepalLengthCm"].plot.hist(alpha=0.5,color='red',bins=50)
# plt.legend(['Iris-setosa','Iris_versicolor','Iris-virginica'])
# plt.xlabel('SepalLengthCm')
# plt.ylabel('Frequency')
# plt.title('SepalLength on Species')
# plt.show()
#
# from sklearn.preprocessing import LabelEncoder
#
# labelencoder = LabelEncoder()
# data["Species"] = labelencoder.fit_transform(data["Species"])
# # data["Species"]
# # Construct a dataframe from a dictionary
# species = pd.DataFrame({'Species': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']})
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

# Sample the train data set while holding out 20% for testing (evaluating) the classifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)
# Features before mean normalization
unscaled_features = X_train

# Mean Normalization (Standarize the features to follow the normal distribution, to obtain a faster & better classifier)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_array = sc.fit_transform(X_train.values) #calculate μ & σ(fit) and apply the transformation(transform)

# Assign the scaled data to a DataFrame & use the index and columns arguments to keep your original indices and column names:
X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
X_test_array = sc.fit_transform(X_test.values)
X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(unscaled_features['SepalLengthCm'], ax=ax1)
sns.kdeplot(unscaled_features['SepalWidthCm'], ax=ax1)
sns.kdeplot(unscaled_features['PetalLengthCm'], ax=ax1)
sns.kdeplot(unscaled_features['PetalWidthCm'], ax=ax1)
ax2.set_title('After Scaling')
sns.kdeplot(X_train['SepalLengthCm'], ax=ax2)
sns.kdeplot(X_train['SepalWidthCm'], ax=ax2)
sns.kdeplot(X_train['PetalLengthCm'], ax=ax2)
sns.kdeplot(X_train['PetalWidthCm'], ax=ax2)
plt.show()
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
mlp = MLPClassifier(hidden_layer_sizes=(4,4,3),solver='sgd',learning_rate_init=0.01,max_iter=500)
mlp.fit(X_train, y_train)
print(mlp.score(X_test,y_test))
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
from sklearn.metrics import accuracy_score
print("Accuracy:%.2f" %accuracy_score(y_test, predictions))
print("Accuracy:%.2f" %mlp.score(X_test_array, y_test))

import numpy as np
# manipulating data via DataFrames, 2-D tabular, column-oriented data structure
import pandas as pd
from pandas import read_excel
from sklearn.manifold import TSNE
from sklearn.svm import SVR
my_sheet = 'FinalAnalysis_forManuscript'
file_name = 'Data_for_ML_crop_residues_assessment.xlsx'
from math import sqrt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
# producing plots and other 2D data visualizations. Use plotly if you want interactive graphs
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
# statistical visualizations (a wrapper around Matplotlib)
import seaborn as sns
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import linear_model
import joblib
from sklearn.manifold import TSNE

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)

# Assign the csv data to a DataFrame
data = pd.read_csv("Data_for_ML_crop_residues_assessment.csv")


# df = read_excel("Data_for_ML_crop_residues_assessment.xlsx", sheet_name = 'FinalAnalysis_forManuscript')
# print(df.head())

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
X1=pd.get_dummies(data, columns=["Code","muname","areasymbol","areaname","Crop Rotations"])
X = X1[['Sand(%)','K_factor','Crop Rotations_CSW','Crop Rotations_CS','Corn Yield(bu/Ha)','Soybean Yield(bu/Ha)','Slope(%)','SlopeLength(m)',
'T_factor','K_factor','Sand(%)','Silt(%)','Clay(%)','Organic matter(%)','Rainfall_Erosivity','Organic Matter factor']]
# X = X1[['Sand(%)','K_factor','Crop Rotations_CSW','Crop Rotations_CS','Corn Yield(bu/Ha)','Soybean Yield(bu/Ha)','Slope(%)','SlopeLength(m)',
# 'T_factor','K_factor','Sand(%)','Silt(%)','Clay(%)','Organic matter(%)','Rainfall_Erosivity','Soil erosion factor']]
# X = X1[['Removal Rate','Corn Yield(bu/Ha)','Clay(%)']]
y = data['Removal Rate']
# y=data['SCI']
# y=data['Soil erosion factor']

corr = X.corr()
    # columns = np.full((corr.shape[0],), True, dtype=bool)
    # for i in range(corr.shape[0]):
    #     for j in range(i + 1, corr.shape[0]):
    #         if corr.iloc[i, j] >= 0.9:
    #             if columns[j]:
    #                 columns[j] = False
    # selected_columns = X.columns[columns]
    # X = X[selected_columns]
# sns.heatmap(corr)
# cor_target = abs(corr["Organic Matter factor"])
# # Selecting highly correlated features
# relevant_features = cor_target[cor_target > 0.3]
# print(relevant_features)

#
# print(data[["Sand(%)", "K_factor"]].corr())
# # print(data[["Sand(%)", "Soil erosion factor"]].corr())

class Stats:

    def __init__(self, X, y, model):
        self.data = X
        self.target = y
        self.model = model
        ## degrees of freedom population dep. variable variance
        self._dft = X.shape[0] - 1
        ## degrees of freedom population error variance
        self._dfe = X.shape[0] - X.shape[1] - 1

    def sse(self):
        '''returns sum of squared errors (model vs actual)'''
        squared_errors = (self.target - self.model.predict(self.data)) ** 2
        return np.sum(squared_errors)

    def sst(self):
        '''returns total sum of squared errors (actual vs avg(actual))'''
        avg_y = np.mean(self.target)
        squared_errors = (self.target - avg_y) ** 2
        return np.sum(squared_errors)

    def r_squared(self):
        '''returns calculated value of r^2'''
        return 1 - self.sse() / self.sst()

    def adj_r_squared(self):
        '''returns calculated value of adjusted r^2'''
        return 1 - (self.sse() / self._dfe) / (self.sst() / self._dft)

def pretty_print_stats(stats_obj):
    '''returns report of statistics for a given model object'''
    items = ( ('sse:', stats_obj.sse()), ('sst:', stats_obj.sst()),
             ('r^2:', stats_obj.r_squared()), ('adj_r^2:', stats_obj.adj_r_squared()) )
    for item in items:
        print('{0:8} {1:.4f}'.format(item[0], item[1]))


# X1 = data[['Code', 'areasymbol']].values
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(categorical_features=[0])
# ohe.fit_transform(X1).toarray()
# data["Code"] = data["Code"].astype('category')
# data["Code"] = data["Code"].cat.codes
# print(data.head())

#
# # Sample the train data set while holding out 20% for testing (evaluating) the classifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
# tsne_results_std = tsne.fit_transform(X_train)
# plt.figure(figsize = (16,11))
# plt.scatter(tsne_results_std[:,0],tsne_results_std[:,1],  c = y_train,
#             cmap = "RdYlGn", edgecolor = "None", alpha=0.35)
# plt.colorbar()
# plt.title('TSNE Scatter Plot')
# plt.show()

# pca = PCA(n_components=14)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))


# scaler.fit(y_train)
#y_train = scaler.transform(y_train)
from sklearn.neighbors import KNeighborsRegressor
# model=LinearRegression()
# model=KNeighborsRegressor()
# model=DecisionTreeRegressor()
model = RandomForestRegressor()
# model=linear_model.Ridge()
# model= linear_model.BayesianRidge()
# model = GradientBoostingRegressor()
# model = GridSearchCV(svm.SVR(),param_grid,refit=True,verbose=2)
# model = MLPRegressor(hidden_layer_sizes=(20,30,60), alpha=1e-8, random_state=1, max_iter=300, warm_start=True,solver='adam', verbose=10, tol=1e-8,
#                     learning_rate_init=.001,activation='relu')
# model=SVR()
model=model.fit(X_train,y_train)
# model.summary()

# joblib.dump(model, "rf_model.pkl")
predictions1= model.predict(X_train)
# model1 = joblib.load('rf_model.pkl')
predictions= model.predict(X_test)
print(X_test.size)
print(y_test.size)


from sklearn.metrics import mean_squared_error


## residuals
# residuals = y_test - predictions
# max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
# max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
# max_true, max_pred = y_test[max_idx], predictions[max_idx]
# print("Max Error:", "{:,.0f}".format(max_error))

fig, ax = plt.subplots()
ax.scatter(predictions, y_test, edgecolors=(0, 0, 1))
ax.plot([predictions.min(), predictions.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()


# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
# tsne_results_std = tsne.fit_transform(X_train)
# plt.figure(figsize = (16,11))
# plt.scatter(tsne_results_std[:,0],tsne_results_std[:,1],  c = y_train,
#             cmap = "RdYlGn", edgecolor = "None", alpha=0.35)
# plt.colorbar()
# plt.title('TSNE Scatter Plot')
# plt.show()
# X_grid = np.arange(min(X_value), max(X_value), 0.01)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X_test, y_test, color = 'red')
# plt.scatter(X_test, predictions, color = 'green')
# plt.title('Random Forest Regression')
# plt.xlabel('Temperature')
# plt.ylabel('Revenue')
# plt.show()

mae = metrics.mean_absolute_error(y_test, predictions)
mae1 = metrics.mean_absolute_error(y_train, predictions1)
print("RMSE",sqrt(mean_squared_error(y_test, predictions)))
print("RMSE Train",sqrt(mean_squared_error(y_train, predictions1)))
mse = metrics.mean_squared_error(y_test, predictions)
mse1 = metrics.mean_squared_error(y_train, predictions1)
r2 = metrics.r2_score(y_test, predictions)
r2_train = metrics.r2_score(y_train, predictions1)
print("MAE",mae)
print("MAE Train",mae1)
print("R2 Train",r2_train)
print("R2",r2)
#
print("MSE",mse)
print("MSE train",mse1)

stats = Stats(X_train, y_train, model)
# pretty_print_stats(stats)



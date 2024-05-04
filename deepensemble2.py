from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from matplotlib import pyplot
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
import seaborn as sns
import numpy as np
from sys import exit
from matplotlib import rcParams
from os import makedirs
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from numpy import dstack
import numpy as np
from xgboost import XGBRegressor
from keras.utils.vis_utils import plot_model

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
import joblib

# wandb.init(project="visualize-sklearn")


from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
data = pd.read_csv("Data_for_ML_crop_residues_assessment.csv")


def get_dataset():
    # X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
    X1 = pd.get_dummies(data, columns=["Code", "muname", "areasymbol", "areaname", "Crop_Rotations"])
    X = X1[['Crop_Rotations_CSW', 'Crop_Rotations_CS', 'Corn Yield(bu/Ha)',
            'Soybean Yield(bu/Ha)', 'Slope(%)', 'SlopeLength(m)',
            'T_factor', 'K_factor', 'Sand(%)', 'Silt(%)', 'Clay(%)', 'Organic matter(%)', 'Rainfall_Erosivity', 'Removal Rate']]

    y = data['Organic Matter factor']

    return X, y


X, y = get_dataset()
print(X.shape)
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler


from sklearn.feature_selection import f_regression

import matplotlib.pyplot as plt
corr_matrix = X.corr().abs()
sns.heatmap(corr_matrix,annot=True,fmt='.1f')
plt.show()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# print(to_drop)
# # Drop features
# X.drop(to_drop, axis=1, inplace=True)
# cor_target = abs(corr["Organic Matter factor"])
# # Selecting highly correlated features
# relevant_features = cor_target[cor_target > 0.3]
# print(relevant_features)

# feature_selector = SelectKBest(f_regression, k = "all")
# fit = feature_selector.fit(X,y)
# p_values = pd.DataFrame(fit.pvalues_)
# scores = pd.DataFrame(fit.scores_)
# input_variable_names = pd.DataFrame(X.columns)
# summary_stats = pd.concat([input_variable_names, p_values, scores], axis = 1)
# print(summary_stats)
# summary_stats.columns = ["input_variable", "p_value", "f_score"]
# summary_stats.sort_values(by = "p_value", inplace = True)
# p_value_threshold = 0.08
# score_threshold = 7
# selected_variables = summary_stats.loc[(summary_stats["f_score"] >= score_threshold) &
#                                        (summary_stats["p_value"] <= p_value_threshold)]
# selected_variables = selected_variables["input_variable"].tolist()
# X_new = X[selected_variables]
# print("Selected columns", X_new.columns)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
print(X_train.shape)

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=mutual_info_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
model1 = Sequential(name="model1_conv1D")
# model.add(keras.layers.Input(shape=(n_timesteps, 16)))
model1.add(Conv1D(64, 2, activation="relu", input_shape=(14, 1)))
model1.add(Conv1D(filters=32, kernel_size=7, activation='relu', name="Conv1D_1"))
# model.add(Dropout(0.5))
model1.add(Conv1D(filters=16, kernel_size=3, activation='relu', name="Conv1D_2"))
# model.add(MaxPooling1D(pool_size=2, name="MaxPooling2D"))
model1.add(Conv1D(filters=32, kernel_size=2, activation='relu', name="Conv1D_4"))
model1.add(MaxPooling1D(pool_size=2, name="MaxPooling1D"))
model1.add(Flatten())
# model.add(Dense(64, activation='relu', name="Dense_1"))
model1.add(Dense(1, name="Dense_2"))
model1.summary()


# models = get_models()
def sse(y, y_pred):
    '''returns sum of squared errors (model vs actual)'''
    squared_errors = (y - y_pred) ** 2
    return np.sum(squared_errors)


def sst(y, y_pred):
    avg_y = np.mean(y)
    squared_errors = (y - avg_y) ** 2
    return np.sum(squared_errors)


def r_squared(sse, sst):
    '''returns calculated value of r^2'''
    return 1 - sse() / sst()


model1.compile(loss='mean_absolute_error', optimizer='adam',metrics=['mean_absolute_error'])
history = model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)
model1.save('deep/model1.h5')
plot_model(model1, to_file='model1_plot.png', show_shapes=True, show_layer_names=True)


model2 = Sequential()
model2.add(Dense(50, activation='relu', input_dim=14))
# model1.add(Dense(25, activation='relu'))
model2.add(Dense(20, activation='relu'))
model2.add(Dense(15, activation='relu'))
model2.add(Dense(10, activation='relu'))
model2.add(Dense(1))

model2.compile(loss='mean_absolute_error', optimizer='adam' )
history1 = model2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)
model2.save('deep/model2.h5')

plot_model(model2, to_file='model2_plot.png', show_shapes=True, show_layer_names=True)

# model3 = GradientBoostingRegressor()
# model3=model3.fit(X_train,y_train)
# joblib.dump(model3, "rf_model.pkl")
# predictions1= model.predict(X_train)
# model3 = Sequential(name="model3_conv1D")
# # model.add(keras.layers.Input(shape=(n_timesteps, 16)))
# model3.add(Conv1D(64, 2, activation="relu", input_shape=(16, 1)))
# model3.add(Conv1D(filters=32, kernel_size=7, activation='relu', name="Conv1D_13"))
# # model.add(Dropout(0.5))
# model3.add(Conv1D(filters=16, kernel_size=3, activation='relu', name="Conv1D_23"))
# # model.add(MaxPooling1D(pool_size=2, name="MaxPooling2D"))
# model3.add(Conv1D(filters=32, kernel_size=2, activation='relu', name="Conv1D_43"))
# model3.add(MaxPooling1D(pool_size=2, name="MaxPooling1D"))
# model3.add(Flatten())
# model3.add(Dense(64, activation='relu', name="Dense_13"))
# model3.add(Dense(1, name="Dense_23"))
# model3.summary()


# model3.compile(loss='mean_absolute_error', optimizer='adam',metrics=['mean_absolute_error'])
# history2 = model3.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
# model3.save('deep/model3.h5')
# plot_model(model3, to_file='model3_plot.png', show_shapes=True, show_layer_names=True)

# model4 = Sequential()
# model4.add(Dense(128, kernel_initializer='normal',activation='relu', input_dim=16))
# model4.add(Dense(256,kernel_initializer='normal', activation='relu'))
# model4.add(Dense(256, kernel_initializer='normal',activation='relu'))
# model4.add(Dense(256, kernel_initializer='normal',activation='relu'))
# model4.add(Dense(1,kernel_initializer='normal',activation='linear'))
#
# model4.compile(loss='mean_absolute_error', optimizer='adam',metrics=['mean_absolute_error'])
# history3 = model4.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)
# model4.save('deep/model4.h5')
# plot_model(model4, to_file='model4_plot.png', show_shapes=True, show_layer_names=True)
dependencies = {
    'r_squared': r_squared
}
# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = './deep/model' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    # model = joblib.load('rf_model.pkl')
    # all_models.append(model)
    return all_models


n_members = 2
members = load_all_models(n_members)
print('Loaded %d models' % len(members))


def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        # yhat = model.predict(inputX, verbose=0)
        yhat = model.predict(inputX)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat  #
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
    return stackX
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # fit standalone model
    model = XGBRegressor() #meta learn
    # model =CatBoostRegressor()
    model.fit(stackedX, inputy)
    return model
model = fit_stacked_model(members, X_test,y_test)

def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)

    return yhat,stackedX

yhat, stackedX = stacked_prediction(members, model, X_test)
score = metrics.r2_score(y_test/1.0, yhat/1.0)
print('Stacked R2 :', score)
mae = metrics.mean_absolute_error(y_test, yhat)
print('MAE of stacked model',mae)

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.n_estimators],
             axis=0)
indices = np.argsort(importances)

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.barh(range(X.shape[1]), importances[indices],
       color="r", xerr=std[indices], align="center")
# If you want to define your own labels,
# change indices to a list of labels on the following line.
plt.yticks(range(X.shape[1]), indices)
plt.ylim([-1, X.shape[1]])
plt.show()


import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
# plt.scatter(y_test,yhat)
# plt.yscale('log')
# plt.xscale('log')



# p1 = max(max(yhat), max(y_test))
# p2 = min(min(yhat), min(y_test))
# plt.plot([p1, p2], [p1, p2], 'b-')
# plt.xlabel('True Values', fontsize=15)
# plt.ylabel('Predictions', fontsize=15)
# plt.axis('equal')
# plt.show()

# visualizer = PredictionError(model)
# # visualizer.fit(X_train, y_train)
# visualizer.score(stackedX, y_test)
# visualizer.show()
test_predictions =yhat.flatten()
fig, ax = plt.subplots()
ax.scatter(test_predictions, y_test, edgecolors=(0, 0, 1))
ax.plot([test_predictions.min(), test_predictions.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Predicted',fontsize=18)
ax.set_ylabel('Actual',fontsize=18)
plt.tick_params(labelsize=16)
plt.show()

error = test_predictions - y_test
print("%Error",error)
plt.hist(error, bins=25)
plt.xlabel('Prediction Error',fontsize=18)
_ = plt.ylabel('Count',fontsize=18)
plt.tick_params(labelsize=16)
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss',fontsize=18)
plt.xlabel('Epoch',fontsize=18)
plt.legend(['Train', 'Val'], fontsize=16)
plt.tick_params(labelsize=16)
plt.show()

i = 0
results, names = list(), list()
for model in members:
    i+=1
    pred = model.predict(X_test)
    score = metrics.r2_score(y_test, pred)
    print('R2-Score of model {} is '.format(i),score)
    results.append(score)

model1=XGBRegressor()
# model1=CatBoostRegressor()
model1=model1.fit(X_train,y_train)
pred1= model1.predict(X_test)
score = metrics.r2_score(y_test, pred1)
print('R2-Score Xgboost is', score)




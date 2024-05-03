# compare machine learning models for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression,RidgeCV
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


data = pd.read_csv("Data_for_ML_crop_residues_assessment.csv")
# get the dataset
def get_dataset():
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
    X1 = pd.get_dummies(data, columns=["Code", "muname", "areasymbol", "areaname", "Crop Rotations"])
    X = X1[['Removal Rate', 'Sand(%)', 'K_factor', 'Crop Rotations_CSW', 'Crop Rotations_CS', 'Corn Yield(bu/Ha)',
            'Soybean Yield(bu/Ha)', 'Slope(%)', 'SlopeLength(m)',
            'T_factor', 'K_factor', 'Sand(%)', 'Silt(%)', 'Clay(%)', 'Organic matter(%)', 'Rainfall_Erosivity']]
    y = data['Soil erosion factor']
    return X, y

# get a list of models to evaluate
def get_stacking():
    # define the base models
    level0 = list()
    level1=list()
    level0.append(('knn', KNeighborsRegressor()))
    level0.append(('mlp',MLPRegressor(hidden_layer_sizes=(20,30,60), alpha=1e-8, random_state=1, max_iter=300, warm_start=True,solver='adam', verbose=10, tol=1e-8,learning_rate_init=.001,activation='relu')))
    level0.append(('cart', DecisionTreeRegressor()))
    level0.append(('svm', SVR()))
    level0.append(('rfr', RandomForestRegressor()))
    level0.append(('lir',LinearRegression()))
    level0.append(('br', linear_model.BayesianRidge()))
    level0.append(('gb', GradientBoostingRegressor()))
    level0.append(('lr', linear_model.LassoCV()))
    # define meta learner model
    level1 = LinearRegression()

    # level1.append((RandomForestRegressor()))
    # level1.append((LogisticRegression()))
    # final_layer = StackingRegressor(
    #     estimators=[('rf', RandomForestRegressor(random_state=42)),
    #                 ('gbrt', GradientBoostingRegressor(random_state=42))],
    #     final_estimator=RidgeCV()
    # )
    # level1 = MLPRegressor(hidden_layer_sizes=(20,30,60), alpha=1e-8, random_state=1, max_iter=300, warm_start=True,solver='adam', verbose=10, tol=1e-8,learning_rate_init=.001,activation='relu')
    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model


# get a list of models to evaluate
def get_models():
    models = dict()
    models['knn'] = KNeighborsRegressor()
    models['mlp'] = MLPRegressor(hidden_layer_sizes=(20,30,60), alpha=1e-8, random_state=1, max_iter=300, warm_start=True,solver='adam', verbose=10, tol=1e-8,learning_rate_init=.001,activation='relu')
    models['cart'] = DecisionTreeRegressor()
    models['svm'] = SVR()
    models['rfr']= RandomForestRegressor()
    models['lir'] = LinearRegression()
    models['br'] = linear_model.BayesianRidge()
    models['gb'] = GradientBoostingRegressor()
    models['lr'] = linear_model.LassoCV()
    models['stacking'] = get_stacking()
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# define dataset
X, y = get_dataset()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = scaler.transform(X)
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
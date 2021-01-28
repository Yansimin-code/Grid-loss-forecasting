#The goal of this notebook is to predict MWh grid loss in power grids using machine learning.
#The dataset is from https://www.kaggle.com/trnderenergikraft/grid-loss-time-series-dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Grouper, DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn import linear_model
from sklearn.svm import LinearSVR, SVR
from sklearn.neural_network  import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from keras.wrappers.scikit_learn import KerasRegressor

#read the dataset
train_raw = pd.read_csv('train.csv', parse_dates=True)
test_raw = pd.read_csv('test_backfilled_missing_data.csv', parse_dates=True)

# Verify shapes are aligned
print(train_raw.shape)
print(test_raw.shape)

#Understanding the data
train_raw.describe()
train_raw.info()

# Look at valid/invalid data of train dataset
train_invalid = train_raw["has incorrect data"][train_raw["has incorrect data"] == True].count()
train_valid = train_raw["has incorrect data"][train_raw["has incorrect data"] == False].count()

print(f"Valid: {train_valid} records")
print(f"Invalid: {train_invalid} records")
print(f"Training data contains {train_invalid/train_valid*100:.2f}% data tagged as invalid")

# Look at valid/invalid data of test dataset
test_invalid = test_raw["has incorrect data"][test_raw["has incorrect data"] == True].count()
test_valid = test_raw["has incorrect data"][test_raw["has incorrect data"] == False].count()

print(f"Valid: {test_valid} records")
print(f"Invalid: {test_invalid} records")
print(f"Test data contains {test_invalid/test_valid*100:.2f}% data tagged as invalid")

# Drop invalid records

train = train_raw[train_raw["has incorrect data"] == False].dropna(subset = ["grid1-load","grid1-loss","grid1-loss-prophet-daily"])
print(train.shape)

test = test_raw[test_raw["has incorrect data"] == False].dropna(subset = ["grid1-load","grid1-loss","grid1-loss-prophet-daily"])

#visualize the grid loss of three grid
ax = train.plot(y=['grid1-loss','grid2-loss','grid3-loss'])
ax.set_xlabel('Time points')
ax.set_ylabel('Grid loss(MWh)')

def regression_results(y_true, y_pred):
    explained_variance = metrics.explained_variance_score(y_true,y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mae,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score

rmse_score = make_scorer(rmse, greater_is_better = False)



def parameter_search_cv(x_train, y_train):

    models_trained = []

    for i, m in enumerate(models):
        name, model = m
        print(f"Training model {name} ...")

        ts_cv = TimeSeriesSplit(n_splits=4)
        #scaler = MinMaxScaler()
        #x_train = scaler.fit_transform(x_train)

        grid_search = GridSearchCV(estimator=model, cv=ts_cv, param_grid=search_params[i], scoring = rmse_score)
        grid_search.fit(x_train, y_train)

        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        best_parameter = grid_search.best_params_

        print(best_model,best_score, best_parameter)

        models_trained.append((name, best_model))

    return models_trained

def cv_result(models_trained, x_train, y_train):

    for name, model in models_trained:
        print("========================================================")
        print(f"Training scores for {name}")
        ts_cv = TimeSeriesSplit(n_splits=4)
        cv_results = cross_val_score(model, x_train, y_train, cv=ts_cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        print(-cv_results)
        print('%s: %f (%f)' % (name, -cv_results.mean(), cv_results.std()))
        print("========================================================")

# Evaluate each model in turn
def predictions(models_trained,x_train, y_train, x_test, y_test):
    predictions = []
    for name, model in models_trained:
        print("========================================================")
        print(f"Test scores for {name}")
        prediction = model.fit(x_train, y_train).predict(x_test)
        testerror=regression_results(y_test, prediction)
        predictions.append(prediction)
# plot a sequence (whole thing is too big)
    plt.plot(y_test[200:600].values, label="True", linewidth=3, alpha=0.7)
    plt.plot(predictions[0][200:600], label="Linear Ridge")
    plt.plot(predictions[1][200:600], label="Random Forest")
    #plt.plot(predictions[2][200:600], label="NN")
    plt.plot(predictions[2][200:600], label="Catboost")
    plt.xlabel("Time points")
    plt.ylabel("MWh")
    plt.legend(loc='upper right')
    plt.rcParams["figure.figsize"] = (30,15)
    plt.rcParams["legend.fontsize"] = 14
    plt.show()
    return predictions

#A two-step method is used to predict the grid loss
#the grid loss is correlated to the grid load, therefore, we built the model to predict the grid load and then predict the grid loss by using the predicted grid load.
#predict load for grid1

grid1load = pd.DataFrame(train, columns=["grid1-load", "grid1-temp","season_x","month_x","week_x","weekday_x","holiday","hour_x"])
grid1load["demand"] = train["demand"]
grid1load["7days_ago"] = grid1load["grid1-load"].shift(7*24)
grid1load["6days_ago"] = grid1load["grid1-load"].shift(6*24)
grid1load = grid1load.dropna()
print(grid1load)

x_train_grid1_load = grid1load.loc[:, grid1load.columns != "grid1-load"]
y_train_grid1_load = grid1load["grid1-load"]

test_grid1load = pd.DataFrame(test, columns=["grid1-load", "grid1-temp","season_x","month_x","week_x","weekday_x","holiday","hour_x"])
test_grid1load["demand"] = test["demand"]
test_grid1load["7days_ago"] = test_grid1load["grid1-load"].shift(7*24)
test_grid1load["6days_ago"] = test_grid1load["grid1-load"].shift(7*24)
test_grid1load = test_grid1load.dropna()

y_test_grid1_load = test_grid1load["grid1-load"]
x_test_grid1_load = test_grid1load.loc[:, test_grid1load.columns != "grid1-load"]

"""# Training"""
# format: tuple("name", ModelClass)
# remember to also add search params
models = [
    ("LR", linear_model.Ridge(max_iter=20000, fit_intercept=True)),
    ("RF", RandomForestRegressor()),
    ("Catboost", CatBoostRegressor(loss_function='MAE', eval_metric="RMSE"))

]

search_params = [
    { 'alpha': [0.01,  0.05,  0.1,  0.5,  1, 2, 5, 10] }, # LR
    { 'n_estimators': [10, 20, 50, 70, 100,200, 500],'max_features': ['sqrt','auto' ], 'max_depth': [i for i in range(5,15)] }, # RF
    {'depth':[3,5,7,8,10], 'iterations':[100,250, 500,1000],'learning_rate':[0.01,0.03,0.1,0.2,0.3] } ,
]


#models_trained_load1 = parameter_search_cv(x_train_grid1_load, y_train_grid1_load)

#to save the time, use the optimum hyperparameter in the following

models_trained_load1 = [
    ("LR", linear_model.Ridge(alpha= 0.01, max_iter=20000, fit_intercept=True)),
    ("RF", RandomForestRegressor(max_depth=11, max_features='sqrt', n_estimators=70)),
    #("NN", MLPRegressor(max_iter=20000)),
    ("Catboost", CatBoostRegressor(depth=7, iterations=1000, learning_rate=0.01, loss_function='MAE', eval_metric="RMSE"))

]

#cv_load1 = cv_result(models_trained_load1, x_train_grid1_load, y_train_grid1_load)
#predictions_load1 = predictions(models_trained_load1,x_train_grid1_load, y_train_grid1_load, x_test_grid1_load, y_test_grid1_load)
#baseline_load_error = regression_results(test_grid1load['grid1-load'],test_grid1load["7days_ago"])
#print(models_trained_load1)


# predit the grid loss for grid1
grid1 = pd.DataFrame(train, columns=["grid1-loss","grid1-temp","season_x","month_x","week_x","weekday_x","holiday","hour_x"])
grid1["6days_ago"] = grid1["grid1-loss"].shift(6*24)
grid1["7days_ago"] = grid1["grid1-loss"].shift(7*24)
grid1["demand"] =  train["demand"]
grid1["prophet_daily"] = train["grid1-loss-prophet-daily"]


grid1 = grid1.dropna()
print(grid1.shape)
print(x_train_grid1_load.shape)
model_load1 = CatBoostRegressor(loss_function='MAE', eval_metric="RMSE", depth=7, learning_rate=0.01, iterations=1000)  # can change the best one here
model_load1.fit(x_train_grid1_load, y_train_grid1_load)
grid1["predicted-load1"] = model_load1.predict(x_train_grid1_load)
#print(grid1["predicted-load1"])
#df = pd.DataFrame(model_load1.predict(x_train_grid1_load), columns=["predicted_load1"])


x_train_grid1 = grid1.loc[:, grid1.columns != "grid1-loss"]
y_train_grid1 = grid1["grid1-loss"]

test_grid1 = pd.DataFrame(test, columns=["grid1-loss", "grid1-temp","season_x","month_x","week_x","weekday_x","holiday","hour_x" ])
test_grid1["6days_ago"] = test_grid1["grid1-loss"].shift(6*24)
test_grid1["7days_ago"] = test_grid1["grid1-loss"].shift(7*24)
test_grid1["demand"] =  test["demand"]
test_grid1["prophet_daily"] = test["grid1-loss-prophet-daily"]
test_grid1 = test_grid1.dropna()
test_grid1["predicted-load1"] = model_load1.predict(x_test_grid1_load)

y_test_grid1 = test_grid1["grid1-loss"]
x_test_grid1 = test_grid1.loc[:, test_grid1.columns != "grid1-loss"]

"""# Training"""


#models_trained_loss = parameter_search_cv(x_train_grid1, y_train_grid1)
models_trained_loss = [
    ("LR", linear_model.Ridge(max_iter=20000, fit_intercept=True, alpha=10)),
    ("RF", RandomForestRegressor(max_depth=5, max_features='sqrt', n_estimators=50, random_state=12)),
    #("NN", MLPRegressor(max_iter=20000)),
    ("Catboost", CatBoostRegressor(loss_function='MAE', eval_metric="RMSE", depth=1, iterations=1000,learning_rate=0.01))

]

baseline_error = regression_results(test_grid1['grid1-loss'],test_grid1['7days_ago'])
print('baseline_error:{0}'.format(baseline_error))
#cv_result(models_trained_loss, x_train_grid1, y_train_grid1)
#predictions(models_trained_loss, x_train_grid1,y_train_grid1,x_test_grid1, y_test_grid1)


# forward feature selection to determine the importance of the feature
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

#for the model of grid load prediction firstly. 
reg_cat_load = CatBoostRegressor(depth=7, iterations=1000, learning_rate=0.01, loss_function='MAE', eval_metric="RMSE")

sfs1 = SFS(reg_cat_load,
           k_features=5,
           forward=True,
           floating=False,
           verbose=2,
           scoring='neg_mean_absolute_error',
           cv=TimeSeriesSplit(n_splits=3))
sfs1 = sfs1.fit(x_train_grid1_load,y_train_grid1_load)


#look at the selected feature indices at each step
print(sfs1.subsets_)

# Which features?
feat_cols_load = list(sfs1.k_feature_idx_)
print(feat_cols_load)

fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdDev)')
plt.xlabel('Number of Features')
plt.ylabel('Negative Mean Absolute Error')
plt.show()

#for the model of grid loss prediction
reg_cat_loss = CatBoostRegressor(loss_function='MAE', eval_metric="RMSE", depth=1, iterations=1000,learning_rate=0.01)

sfs2 = SFS(reg_cat_loss,
           k_features=5,
           forward=True,
           floating=False,
           verbose=2,
           scoring='neg_mean_absolute_error',
           cv=TimeSeriesSplit(n_splits=3))

sfs2 = sfs2.fit(x_train_grid1, y_train_grid1)

#look at the selected feature indices at each step
print(sfs2.subsets_)

# Which features?
feat_cols_loss = list(sfs2.k_feature_idx_)
print(feat_cols_loss)

fig2 = plot_sfs(sfs2.get_metric_dict(), kind='std_err')

#plt.title('Sequential Forward Selection (w. StdDev)')
plt.xlabel('Number of Features')
plt.ylabel('Negative Mean Absolute Error')

#plt.grid()
plt.show()




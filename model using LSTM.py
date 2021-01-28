#Long Short Time Memory is used to predict the grid loss

import pandas as pd
import numpy as np
from numpy import concatenate
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Grouper, DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from catboost import CatBoostRegressor
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from keras.wrappers.scikit_learn import KerasRegressor

train_raw = pd.read_csv('train.csv', parse_dates=True)
test_raw = pd.read_csv('test_backfilled_missing_data.csv', parse_dates=True)

# Verify shapes are aligned
print(train_raw.shape)
print(test_raw.shape)

train_raw.describe()

train_raw.info()

# Look at valid/invalid data
train_invalid = train_raw["has incorrect data"][train_raw["has incorrect data"] == True].count()
train_valid = train_raw["has incorrect data"][train_raw["has incorrect data"] == False].count()

print(f"Valid: {train_valid} records")
print(f"Invalid: {train_invalid} records")
print(f"Training data contains {train_invalid/train_valid*100:.2f}% data tagged as invalid")

# Look at valid/invalid data
test_invalid = test_raw["has incorrect data"][test_raw["has incorrect data"] == True].count()
test_valid = test_raw["has incorrect data"][test_raw["has incorrect data"] == False].count()

print(f"Valid: {test_valid} records")
print(f"Invalid: {test_invalid} records")
print(f"Test data contains {test_invalid/test_valid*100:.2f}% data tagged as invalid")

# Drop invalid records
train = train_raw[train_raw["has incorrect data"] == False].dropna(subset = ["grid1-load","grid1-loss","grid1-loss-prophet-daily"])
print(train.shape)

test = test_raw[test_raw["has incorrect data"] == False].dropna(subset = ["grid1-load","grid1-loss","grid1-loss-prophet-daily"])


# predict load for grid1

grid1load = pd.DataFrame(train, columns=["grid1-load", "grid1-temp","season_x","month_x","week_x","weekday_x","holiday","hour_x"])
grid1load["demand"] = train["demand"]
grid1load["6days_ago"] = grid1load["grid1-load"].shift(6*24)
grid1load["7days_ago"] = grid1load["grid1-load"].shift(7*24)
grid1load = grid1load.dropna()
print(grid1load)

x_train_grid1_load = grid1load.loc[:, grid1load.columns != "grid1-load"]
y_train_grid1_load = grid1load["grid1-load"]

test_grid1load = pd.DataFrame(test, columns=["grid1-load", "grid1-temp","season_x","month_x","week_x","weekday_x","holiday","hour_x"])
test_grid1load["7days_ago"] = test_grid1load["grid1-load"].shift(7*24)
test_grid1load["demand"] = test["demand"]
test_grid1load["6days_ago"] = test_grid1load["grid1-load"].shift(6*24)
test_grid1load = test_grid1load.dropna()

y_test_grid1_load = test_grid1load["grid1-load"]
x_test_grid1_load = test_grid1load.loc[:, test_grid1load.columns != "grid1-load"]


# predit the grid loss for grid1
grid1 = pd.DataFrame(train, columns=["grid1-loss","grid1-temp","season_x","month_x","week_x","weekday_x","holiday","hour_x"])
grid1["7days_ago"] = grid1["grid1-loss"].shift(7*24)
grid1["6days_ago"] = grid1["grid1-loss"].shift(6*24)
grid1["demand"] =  train["demand"]
grid1["prophet_daily"] = train["grid1-loss-prophet-daily"]
grid1 = grid1.dropna()
print(grid1)

#here the optimum parameter is used. This is obtain from the file named two step forescasting.py 
model_load1 = CatBoostRegressor(loss_function='MAE', eval_metric="RMSE", depth=5, learning_rate=0.01, iterations=1000)  # can change the best one here
model_load1.fit(x_train_grid1_load, y_train_grid1_load)
grid1["predicted-load1"] = model_load1.predict(x_train_grid1_load)
#print(grid1["predicted-load1"])
#df = pd.DataFrame(model_load1.predict(x_train_grid1_load), columns=["predicted_load1"])



test_grid1 = pd.DataFrame(test, columns=["grid1-loss", "grid1-temp","season_x","month_x","week_x","weekday_x","holiday","hour_x" ])
test_grid1["7days_ago"] = test_grid1["grid1-loss"].shift(7*24)
test_grid1["6days_ago"] = test_grid1["grid1-loss"].shift(6*24)
test_grid1["demand"] =  test["demand"]
test_grid1["prophet_daily"] = test["grid1-loss-prophet-daily"]
test_grid1 = test_grid1.dropna()
test_grid1["predicted_load"] = model_load1.predict(x_test_grid1_load)


"""# Training"""

#for LSTM it need species formate of input, which includes a time step and scale of the data
#trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

#scale train and test data to [0,1] since relu activation function is used in LSTM
def scale(train, test):
    #fit scale
    scaler = MinMaxScaler(feature_range =(0,1))
    scaler = scaler.fit(train)
    #train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    #test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

#inverse scaling for a forecasted value
def invert_scale(scaler, X):
    inverse_transformed(X)
    return inverted


#define LSTM neural network
def creat_lstmmodel(n_nodes=1,n_input=1):
    model_lstm = Sequential()
    model_lstm.add(LSTM(n_nodes, activation = 'relu', input_shape = (n_input, 1)))
    model_lstm.add(Dense(n_nodes, activation = 'relu'))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mse', optimizer = 'adam')
    return model_lstm


x_train_grid1 = grid1.iloc[:, 1:]
x_test_grid1_invert = test_grid1.iloc[:, 1:]

scaler_x, x_train_grid1, x_test_grid1 = scale(x_train_grid1, x_test_grid1_invert)

y_train_grid1 = grid1.iloc[:,0].values.reshape(-1,1)
y_test_grid1_invert = test_grid1.iloc[:,0].values.reshape(-1,1)

scaler_y, y_train_grid1, y_test_grid1 = scale(y_train_grid1, y_test_grid1_invert)

time_step = 7*24
num_features = x_train_grid1.shape[1]
nb_samples = x_train_grid1.shape[0]-time_step

x_train_reshaped = np.zeros((nb_samples, time_step, num_features))
y_train_reshaped = np.zeros((nb_samples))

nb_samples_test = x_test_grid1.shape[0]-time_step
x_test_reshaped = np.zeros((nb_samples_test, time_step, num_features))
y_test_reshaped = np.zeros((nb_samples_test))


for i in range(nb_samples):
    y_position = i + time_step
    x_train_reshaped[i] = x_train_grid1[i:y_position]
    y_train_reshaped[i] = y_train_grid1[y_position]

for i in range(nb_samples_test):
    y_position = i + time_step
    x_test_reshaped[i] = x_test_grid1[i:y_position]
    y_test_reshaped[i] = y_test_grid1[y_position]

print(x_test_reshaped)


model = Sequential()
model.add(LSTM(100,  input_shape=(x_train_reshaped.shape[1], x_train_reshaped.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer = 'adam')

history = model.fit(x_train_reshaped, y_train_reshaped, epochs=100, batch_size=100)
#plt.plot(history.history['loss'], label='train')
#plt.legend()


#make a prediction
y_predict = model.predict(x_test_reshaped)

#invert scaling for forecast and actual

inv_y_predict = scaler_y.inverse_transform(y_predict)
print(inv_y_predict)


regression_results(y_test_grid1_invert[time_step:], inv_y_predict)

plt.plot(y_test_grid1_invert[200+time_step:600+time_step],label="true", linewidth = 3, alpha = 0.7)
plt.plot(inv_y_predict[200:600], label = "LSTM")
plt.legend(loc='upper right')
plt.xlabel("Time Points")
plt.ylabel("Grid1-loss(KWh)")
plt.rcParams["figure.figsize"] = (30,15)
plt.show()

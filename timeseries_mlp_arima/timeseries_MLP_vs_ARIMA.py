# TIME SERIES PREDICTIONS: MLP NETWORK VS ROLLING FORECAST ARIMA

%cd '/Users/kseniya/github/python_projects/timeseries analysis'
%pwd


import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from math import sqrt
import matplotlib
from matplotlib import pyplot
import numpy 
import moment

# Reading and Preprocessing

# Datetime parsing function 
def parser(x):
    m = moment.date(x, "%Y-%m-%d %H:%M:%S+00:00")
    return m.format('YYYY-M-D H:M')

# Read 4 CSV-files (Spark product)
series1 = read_csv('datasets/1.csv', header=0, parse_dates=[1], index_col=0, squeeze=True, date_parser=parser)
series2 = read_csv('datasets/2.csv', header=0, parse_dates=[1], index_col=0, squeeze=True, date_parser=parser)
series3 = read_csv('datasets/3.csv', header=0, parse_dates=[1], index_col=0, squeeze=True, date_parser=parser)
series4 = read_csv('datasets/4.csv', header=0, parse_dates=[1], index_col=0, squeeze=True, date_parser=parser)

# Leave required columns
df1 = series1[['DateTime', 'Direction', 'Latency', 'NetworkGeneration', 'NetworkOperatorName']].copy()
df2 = series2[['DateTime', 'Direction', 'Latency', 'NetworkGeneration', 'NetworkOperatorName']].copy()
df3 = series3[['DateTime', 'Direction', 'Latency', 'NetworkGeneration', 'NetworkOperatorName']].copy()
df4 = series4[['DateTime', 'Direction', 'Latency', 'NetworkGeneration', 'NetworkOperatorName']].copy()

# Create one resulting DataFrame filtered for required values
frames = [df1, df2, df3, df4]
result = pd.concat(frames)
df = result.ix[(result['Direction'] == 'Download') & 
               (result['NetworkGeneration'] == '4G') & 
               (result['NetworkOperatorName'] == 'Verizon Wireless')]
df.reset_index(inplace=True)
df.drop(['CellTowerId'], axis=1, inplace = True)

# Resample DF to make equal time periods
df_resampled = df.resample('60min', on='DateTime').mean()

# Get statistics
df_resampled.describe()

# Count NaN's, abour 45% of values are NaN's
df_resampled.isnull().sum()

# Removing rows with missing values can be too limiting on some predictive 
# modeling problems, an alternative is to impute missing values
# Possible extension: Tackle how it limits our models

# Imputing missing values with mean()
df_resampled.fillna(df_resampled.mean(), inplace=True)


# Part 1 - Multilayer Perceptron Model
# Courtesy - Jason Brownlee

# Frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	return df

# Create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# Invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# Scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# Inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
	new_row = [x for x in X] + [yhat]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# Fit an MLP network to training data
def fit_model(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	model = Sequential()
	model.add(Dense(neurons, activation='relu', input_dim=X.shape[1]))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=0, shuffle=False)
	return model

# Run a repeated experiment
def experiment(repeats, series, epochs, lag, neurons):
	# Transform data to be stationary
	raw_values = series.values
	diff_values = difference(raw_values, 1)
	# Transform data to be supervised learning
	supervised = timeseries_to_supervised(diff_values, lag)
	supervised_values = supervised.values[lag:,:]
	# Split data into train and test-sets
	train, test = supervised_values[0:-315], supervised_values[-315:]
	# Transform the scale of the data
	scaler, train_scaled, test_scaled = scale(train, test)
	# Run experiment
	error_scores = list()
	for r in range(repeats):
		# Fit the model
		batch_size = 4
		train_trimmed = train_scaled[2:, :]
		model = fit_model(train_trimmed, batch_size, epochs, neurons)
		# Forecast test dataset
		test_reshaped = test_scaled[:,0:-1]
		output = model.predict(test_reshaped, batch_size=batch_size)
		predictions = list()
		for i in range(len(output)):
			yhat = output[i,0]
			X = test_scaled[i, 0:-1]
			# Invert scaling
			yhat = invert_scale(scaler, X, yhat)
			# Invert differencing
			yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
			# Store forecast
			predictions.append(yhat)
		# Report performance
		rmse = sqrt(mean_squared_error(raw_values[-315:], predictions))
		print('%d) Test RMSE: %.3f' % (r+1, rmse))
		error_scores.append(rmse)
	return error_scores, predictions, numpy.asarray(raw_values[-315:], dtype=numpy.float32)

# Set parameters for experiment
repeats = 1
results = DataFrame(['Error', 'Predictions', 'Actual'])
lag = 1
neurons = 1
epochs = 20

# Report results
results = experiment(repeats, df_resampled, epochs, lag, neurons)



# Part 2 - Rolling Forecast with ARIMA model

# Monkey patch around bug in ARIMA class
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
ARIMA.__getnewargs__ = __getnewargs__

# Create Rolling Forecast ARIMA model for our data
# (p, q, d) - values (AR, I, MA) for simplicty taken from the example
# But, for further tuning Akaike's Information Criterion (AIC) or Schwartz Bayesian Information Criterion (BIC)
# should be used - the lowest the better 
# R has auto.arima function, library(forecast). For example, auto.arima(x, ic = "aic")
series_arima = df_resampled
X = series_arima.values
size = int(len(X) * 0.8)
train_arima, test_arima = X[0:size], X[size:len(X)]
history = [x for x in train_arima]
predictions = list()
for t in range(len(test_arima)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)    
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_arima[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
RMSE_arima = sqrt(mean_squared_error(test_arima, predictions))
print('Test RMSE ARIMA: %.3f' % RMSE_arima)

# Save model
model_fit.save('arima_model.pkl')
# Save data
numpy.save('arima_data.npy', X)
# Save the last observation
numpy.save('arima_obs.npy', [series_arima.values[-1]])  

# Plot resulting predictions vs actual test set data (validation dataset)
pyplot.plot(test_arima)
pyplot.plot(predictions, color='red')
pyplot.show()


# Part 3 - Predictions with the chosen approach
# Based on RMSE estimations for two simple models, ARIMA performed better
# Next, we deliver predictions for the next 24 hours 

# Load the saved model from file
import numpy
model = ARIMAResults.load('arima_model.pkl')
data = numpy.load('arima_data.npy')
print(data)
last_ob = numpy.load('arima_obs.npy')
print(last_ob)
list_forecast = list()

for i in range(24):
    model = ARIMAResults.load('arima_model.pkl')
    data = numpy.load('arima_data.npy')
    last_ob = numpy.load('arima_obs.npy')
    # Make prediction
    forecast = model.predict(start=len(data), end=len(data))
    forecasting = forecast[0] + last_ob[0]
    print('Prediction: %f' % forecasting)
    list_forecast = list_forecast + forecasting

    # Update data with one rolling forecast value
    forecasting = forecasting.reshape(1,1)
    data = numpy.append(data, forecasting, axis=0)
    numpy.save('arima_data.npy', data)

    # Update and save the last delivered observation
    last_ob[0] = forecasting
    numpy.save('arima_obs.npy', last_ob)
    
    i += 1
     
     
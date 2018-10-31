# Time series predictions: MLP network vs Rolling Forecast ARIMA

## Problem Description
There are four different downloads from Spark in the format of CSV-files. 
The task included integration of data, filtering and choosing a model for predictions.
The data represents a time series with uneven time periods.

## Prerequisits
* Python IDE
* Tensorflow 
* Keras
* Pandas, Numpy, Matplotlib, Moment, Statsmodels packages 

## My Solution
### MLP
Multilayer Perceptron (MLP) is a classical type of neural networks. May comprise one or more layers of neurons. 
Data is fed to the input layers, and predictions are made on the output layer while network
itself can have a few hidden layers beside visible one.
MLP performs well on tabular format data, solving both classification and regression prediction problems.
The lag observations for a time series prediction can be presented as a row of data and also
serve as an input to the model. MLP can be a good comparison model to test other models.

### ARIMA
AutoRegressive Integrated Moving Average (ARIMA). 
Auto Regressive (AR) terms refer to the lags of the differenced series, 
Moving Average (MA) terms refer to the lags of errors and I is the number of difference used to make the time series stationary.
The ARIMA model is well suited to work with time series data and can be used to forecast future time steps.

Assumptions:
* Data should be stationary – by stationary it means that the properties of the series doesn’t depend on the time when it is captured. A white noise series and series with cyclic behavior can also be considered as stationary series.
* Data should be univariate – ARIMA works on a single variable. Auto-regression is all about regression with the past values.

### Approach
For MLP, I take a classical approach of splitting the data into train and test sets. 
Test set will serve as a validation and the modelling will be performed in accordance with rolling-forecast scenario
or walk-forward model validation. Each time step of a test data will be analyzed one at a time. 
All predicted values will be collected and error metric will be calculated. 
MLP represents a very basic model with 1 neuron hidden layer, rectified linear activation function on hidden layer, 
and linear activation function on the output. I also use the efficient ADAM optimizer. 

The rolling forecast approach will be implemented with ARIMA as well. Forecast() function allows to perform 1-step forecast by the model.
With high dependence on prior observations for differencing and Auto-Regression model, the rolling forecast is required.
I also re-create the model after each new value being received, which might be not so elegant, but still implemented
to keep focus on more important elements of the project. 

As for (p,d,q) order of the model, plotting data or running  allows to tune. 
However, for further tuning Akaike's Information Criterion (AIC) or Schwartz Bayesian Information Criterion (BIC) 
should be used - the lowest value for a model the better. R has auto.arima function, library(forecast). For example, auto.arima(x, ic = "aic").
Because it's not the focus of the project, so we choose lag value of 5 for AR, difference order 1 for the stationarity, I, and 0 
for MA model.

### Point of comparison
For both models RMSE (standard deviation of the prediction errors) is used as it delivers results in the same units as the forecast data.
It's implemented with importing MSE from scikit-learn metrics library and taking a square root of it. 

### Results
There are contradictory opinions on performance of ARIMA and Neural Networks on time series forecasting problems.
In my case, ARIMA delivers comparatively lower RMSE in juxtaposition to the basic MLP. However, I should point out
once more there have been no well-performed tuning done for any of two presented models. As a end-point for my project,
I save the model and its parameters and perform a rolling forecast for the next 24 hours while updating data and last observation variable for each step.

The purpose of this limited-time project was 
* exploration of two different methods of time series analysis, their preliminary comparison
based on RMSE metric  
* establishment of some grounds to further thoughts that in some cases we might not need complex architectures
to solve time series prediction problems, and employment of well-established approaches, such as ARIMA, might do the trick with better accuracy of forecast data.

## Further readings
As an extension, I would like to recommend the paper on the comparison of performances of ARIMA and ANN model based on NY Stock Exchange Data -
Ayodele Ariyo Adebiyi, Aderemi Oluyinka Adewumi, and Charles Korede Ayo, “Comparison of ARIMA and Artificial Neural Networks Models for Stock Price Prediction,” Journal of Applied Mathematics, vol. 2014, Article ID 614342, 7 pages, 2014. https://doi.org/10.1155/2014/614342.

## Acknowledgments
* Jason Brownlee, PhD for his insightful and well-written articles on TS problems for ML engineers 








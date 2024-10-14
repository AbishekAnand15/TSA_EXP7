# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Read the CSV file into a DataFrame (replace with the path to the sales data)
data = pd.read_csv("/content/Super_Store_data.csv",encoding='ISO-8859-1')  

# Parse the date and set it as index
data['Order Date'] = pd.to_datetime(data['Order Date'])
data.set_index('Order Date', inplace=True)

# Filter for 'Furniture' sales
furniture_sales = data[data['Category'] == 'Furniture']

# Aggregate monthly sales
monthly_sales = furniture_sales['Sales'].resample('MS').sum()

# Perform Augmented Dickey-Fuller test to check stationarity
result = adfuller(monthly_sales) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets (80% train, 20% test)
train_data = monthly_sales.iloc[:int(0.8*len(monthly_sales))]
test_data = monthly_sales.iloc[int(0.8*len(monthly_sales)):]

# Fit an AutoRegressive (AR) model with 13 lags
lag_order = 13
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

# Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
plot_acf(monthly_sales)
plt.title('Autocorrelation Function (ACF) - Furniture Sales')
plt.show()

plot_pacf(monthly_sales)
plt.title('Partial Autocorrelation Function (PACF) - Furniture Sales')
plt.show()

# Make predictions using the AR model
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

# Compare the predictions with the test data
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)

# Plot the test data and predictions
plt.plot(test_data.index, test_data, label='Test Data', color='blue')
plt.plot(test_data.index, predictions, label='Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('AR Model Predictions vs Test Data - Furniture Sales')
plt.legend()
plt.show()

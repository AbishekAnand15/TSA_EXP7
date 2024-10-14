# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
#### Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

#### Read the CSV file into a DataFrame
data = pd.read_csv("/content/superstore_sales.csv")  

#### Parse the date and set it as index
data['Order Date'] = pd.to_datetime(data['Order Date'])
data.set_index('Order Date', inplace=True)

#### Filter for 'Furniture' sales
furniture_sales = data[data['Category'] == 'Furniture']

#### Aggregate monthly sales
monthly_sales = furniture_sales['Sales'].resample('MS').sum()

#### Perform Augmented Dickey-Fuller test to check stationarity
result = adfuller(monthly_sales) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])

#### Split the data into training and testing sets
train_data = monthly_sales.iloc[:int(0.8*len(monthly_sales))]
test_data = monthly_sales.iloc[int(0.8*len(monthly_sales)):]

#### Fit an AutoRegressive (AR) model with 13 lags
lag_order = 13
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

#### Plot Autocorrelation Function (ACF)
plot_acf(monthly_sales)
plt.title('Autocorrelation Function (ACF) - Furniture Sales')
plt.show()

#### Plot Partial Autocorrelation Function (PACF)
plot_pacf(monthly_sales)
plt.title('Partial Autocorrelation Function (PACF) - Furniture Sales')
plt.show()

#### Make predictions using the AR model
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

#### Compare the predictions with the test data
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)

#### Plot the test data and predictions
plt.plot(test_data.index, test_data, label='Test Data', color='blue')
plt.plot(test_data.index, predictions, label='Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('AR Model Predictions vs Test Data - Furniture Sales')
plt.legend()
plt.show()

### OUTPUT:

#### GIVEN DATA
![image](https://github.com/manojvenaram/TSA_EXP7/assets/94165064/55464c15-0b8e-494a-9403-a83ff3cdf99a)
#### Augmented Dickey-Fuller test
![image](https://github.com/manojvenaram/TSA_EXP7/assets/94165064/a35cd199-a5c7-48af-94f7-e9876e266333)

#### PACF - ACF
![image](https://github.com/manojvenaram/TSA_EXP7/assets/94165064/836855ba-78bf-4dae-b447-bfe0212e99b3)
![image](https://github.com/manojvenaram/TSA_EXP7/assets/94165064/bbb16425-6806-4f62-90c9-121341f09106)
#### Mean Squared Error
![image](https://github.com/manojvenaram/TSA_EXP7/assets/94165064/e22306cf-9799-48dd-ad79-8ea4861a2e01)
#### PREDICTION
![image](https://github.com/manojvenaram/TSA_EXP7/assets/94165064/9749b788-f6c8-4309-a735-b7e8146509b1)


### RESULT:
Thus we have successfully implemented the auto regression function using python.

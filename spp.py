import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

if __name__ == '__main__':
    """Load training data set with the "Open" and "High" columns to use in our modeling."""

    #url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
    dataset_train = pd.read_csv('NSE-TATAGLOBAL.csv')
    training_set = dataset_train.iloc[:, 1:2].values

    """Let's take a look at the first five rows of our dataset"""
    print("\n||||||||||||||||||||||||||||FIRST 5 ROWS OF THE DATASET||||||||||||||||||||||||||\n")
    print(dataset_train.head())

    """Import MinMaxScaler from scikit-learn 
    to scale our dataset into numbers between 0 and 1 """

    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    """We want our data to be in the form of a 3D array for our LSTM model. 
    First, we create data in 60 timesteps and convert it into an array using NumPy. 
    Then, we convert the data into a 3D array with X_train samples, 60 timestamps, 
    and one feature at each step."""
    X_train = []
    y_train = []
    for i in range(60, 2035):
        X_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Make the necessary imports from keras
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.layers import Dense

    """Add LSTM layer along with dropout layers to prevent overfitting. 
    After that, we add a Dense layer that specifies a one unit output.
    Next, we compile the model using the adam optimizer and set the loss as the mean_squarred_error"""

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=10, batch_size=32)

    """Import the test set for the model to make predictions on """

    #url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv'
    dataset_test = pd.read_csv('tatatest.csv')
    print("\n|||||||||||||||||||||||||FIRST 5 ROWS OF DATASET USED FOR TESTING|||||||||||||||||||||||||\n", dataset_test.head())
    real_stock_price = dataset_test.iloc[:, 1:2].values

    """
    Before predicting future stock prices, we have to manipulate the training set; 
    we merge the training set and the test set on the 0 axis,
     set the time step to 60, use minmaxscaler, 
     and reshape the dataset as done previously. After making predictions,
      we use inverse_transform to get back the stock prices in normal readable format.
    """

    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 76):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    """Plot our predicted stock prices and the actual stock price"""

    plt.plot(real_stock_price, color='black', label='TATA Stock Price')
    plt.plot(predicted_stock_price, color='green', label='Predicted TATA Stock Price')
    plt.title('TATA Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('TATA Stock Price')
    plt.legend()
    plt.show()


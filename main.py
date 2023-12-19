import streamlit as st
from datetime import date
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Function to load data
@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime
    data.set_index('Date', inplace=True)  # Set 'Date' as the index
    return data

# Function to plot raw data
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Function to create and train a simple neural network
def create_and_train_model(data, years=3):
    # Extract the past 'years' years of data
    start_date = TODAY - pd.DateOffset(years=years)
    training_data = data[data.index >= start_date]

    df_train = training_data[['Close']]
    df_train = df_train.rename(columns={"Close": "y"})

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_train['y'].values.reshape(-1, 1))

    # Use the past 60 days of data for training
    training_data = scaled_data[-60:]

    # Create scaled training data set
    x_train, y_train = [], []

    for i in range(1, len(training_data)):
        x_train.append(training_data[i - 1:i, 0])
        y_train.append(training_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model using the past 60 days of data
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    return model, scaler, scaled_data  # Remove training_data_len

# Function to calculate MAPE
def calculate_mape(actual_values, predicted_values):
    return np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

# Function to create and plot the forecast
def create_and_plot_forecast(model, scaler, scaled_data, data):
    # Use the last 1 day of data for prediction
    last_day = scaled_data[-1:]

    x_test = np.array(last_day)
    x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))

    # Predict the values for the next day
    predicted_values = model.predict(x_test)
    predicted_values = scaler.inverse_transform(predicted_values.reshape(-1, 1))

    # Calculate the daily percentage change
    daily_change = predicted_values.flatten() / data['Close'].iloc[-1]

    # Get the last known values
    last_open = data['Open'].iloc[-1]
    last_high = data['High'].iloc[-1]
    last_low = data['Low'].iloc[-1]

    # Predicted values for the next day
    predicted_open = last_open * daily_change
    predicted_high = last_high * daily_change
    predicted_low = last_low * daily_change

    # Create a new DataFrame with the next day's date
    next_day_date = data.index[-1] + pd.DateOffset(1)
    next_day_df = pd.DataFrame(index=[next_day_date])

    # Add the predicted values to the DataFrame
    next_day_df['Predicted Close'] = predicted_values.flatten()
    next_day_df['Predicted Open'] = predicted_open
    next_day_df['Predicted High'] = predicted_high
    next_day_df['Predicted Low'] = predicted_low
    next_day_df['Predicted Volume'] = data['Volume'].iloc[-1] * daily_change

    # Calculate Mean Absolute Percentage Error (MAPE)
    actual_values = data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1].values
    predicted_values = next_day_df[['Predicted Open', 'Predicted High', 'Predicted Low', 'Predicted Close', 'Predicted Volume']].values

    mape = calculate_mape(actual_values, predicted_values)

    # Append MAPE to the raw data table
    data.loc[next_day_date, 'MAPE'] = mape

    # Plot the results
    st.subheader('Forecast data for the next day')
    st.write(next_day_df)

    # Plot the forecast
    st.write('Forecast plot for the next day')
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical Data'))
    fig.add_trace(go.Scatter(x=next_day_df.index, y=next_day_df['Predicted Close'], name='Predicted Close'))
    fig.add_trace(go.Scatter(x=next_day_df.index, y=next_day_df['Predicted Open'], name='Predicted Open'))
    fig.add_trace(go.Scatter(x=next_day_df.index, y=next_day_df['Predicted High'], name='Predicted High'))

    fig.layout.update(title_text='Time Series Forecast for the Next Day', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    # Display MAPE
    st.subheader('Model Accuracy (Mean Absolute Percentage Error)')
    st.write(f'MAPE: {mape:.2f}%')

# Main Streamlit app code
START = "2015-01-01"
TODAY = pd.Timestamp(date.today().strftime("%Y-%m-%d"))

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'HD', 'WMT')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
plot_raw_data(data)

# Create and train the neural network model using the past 3 years of data
model, scaler, scaled_data = create_and_train_model(data, years=3)

# Create and plot the forecast for the next day
create_and_plot_forecast(model, scaler, scaled_data, data)

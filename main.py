# pip install streamlit yfinance plotly scikit-learn
import streamlit as st
from datetime import date
import pandas as pd

import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'NVDA', 'WMT', 'HD')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Predict forecast with Linear Regression (replace with other models as needed)
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Create features (you might need to engineer more features)
df_train['ds'] = pd.to_datetime(df_train['ds'])
df_train['ds_numeric'] = df_train['ds'].astype(np.int64) // 10**9

# Linear Regression model
model = LinearRegression()
model.fit(df_train[['ds_numeric']], df_train['y'])

# Generate future dates
future_dates = pd.date_range(start=df_train['ds'].max(), periods=period, freq='D')
future_numeric = (future_dates.astype(np.int64) // 10**9).values.reshape(-1, 1)

# Predict future values
forecast_values = model.predict(future_numeric)

# Create a DataFrame for forecast
forecast = pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines', name='Historical Data'))
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
fig1.layout.update(title_text='Time Series Forecast with Linear Regression', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

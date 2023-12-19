# pip install streamlit yfinance plotly scikit-learn
import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import date
from sklearn.linear_model import LinearRegression

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

# Plot raw data with bars for daily prices
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data['Date'], y=data['Open'].tolist(), name="Open"))
    fig.add_trace(go.Bar(x=data['Date'], y=data['Close'].tolist(), name="Close"))

    # Add flags for highs and lows
    high_flags = go.Scatter(
        x=data['Date'][data['High'] == data['High'].max()],
        y=[data['High'].max()],  # Convert to a list
        mode='markers',
        name='High',
        marker=dict(color='red', size=8),
        text='High'
    )

    low_flags = go.Scatter(
        x=data['Date'][data['Low'] == data['Low'].min()],
        y=[data['Low'].min()],  # Convert to a list
        mode='markers',
        name='Low',
        marker=dict(color='blue', size=8),
        text='Low'
    )

    fig.add_trace(high_flags)
    fig.add_trace(low_flags)

    fig.layout.update(title_text='Time Series data with Bars and Flags', xaxis_rangeslider_visible=True)
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

# Show and plot forecast with bars
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = go.Figure()

# Convert 'y' values to a list for both historical data and forecast
fig1.add_trace(go.Bar(x=data['Date'], y=data['Close'].tolist(), name='Historical Data', opacity=0.6))
fig1.add_trace(go.Bar(x=forecast['ds'], y=forecast['yhat'].tolist(), name='Forecast', opacity=0.6))

# Add flags for forecasted highs and lows
forecast_high_flag = go.Scatter(
    x=forecast['ds'][forecast['yhat'] == forecast['yhat'].max()],
    y=[forecast['yhat'].max()],  # Convert to a list
    mode='markers',
    name='Forecast High',
    marker=dict(color='red', size=8),
    text='Forecast High'
)

forecast_low_flag = go.Scatter(
    x=forecast['ds'][forecast['yhat'] == forecast['yhat'].min()],
    y=[forecast['yhat'].min()],  # Convert to a list
    mode='markers',
    name='Forecast Low',
    marker=dict(color='blue', size=8),
    text='Forecast Low'
)

fig1.add_trace(forecast_high_flag)
fig1.add_trace(forecast_low_flag)

fig1.layout.update(title_text='Time Series Forecast with Linear Regression and Bars', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

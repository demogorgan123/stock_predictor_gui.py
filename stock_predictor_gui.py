import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

def load_data(ticker="AAPL", period="5y"):
    df = yf.download(ticker, period=period)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    return df

def prepare_data(data, feature='Close', sequence_len=60):
    values = data[feature].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    X, y = [], []
    for i in range(sequence_len, len(scaled)):
        X.append(scaled[i-sequence_len:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_predict(model, X_train, y_train, X_test, scaler):
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions)

def plot_stock(df):
    st.subheader("Stock Price and Moving Average")
    plt.figure(figsize=(14, 6))
    plt.plot(df['Close'], label='Close')
    plt.plot(df['MA20'], label='20-Day MA')
    plt.legend()
    st.pyplot(plt)

    st.subheader("Volatility")
    plt.figure(figsize=(14, 4))
    plt.plot(df['Volatility'], label='20-Day Volatility', color='purple')
    plt.legend()
    st.pyplot(plt)

def plot_prediction(df, actual, predicted, split_idx):
    st.subheader("Prediction vs Actual")
    plt.figure(figsize=(14, 6))
    plt.plot(df.index[split_idx:], actual, label='Actual', color='blue')
    plt.plot(df.index[split_idx:], predicted, label='Predicted', color='red')
    plt.legend()
    st.pyplot(plt)

st.title("📈 Stock Market Prediction Tool with AI")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", value="AAPL")
period = st.selectbox("Select Time Period", ["1y", "2y", "5y"], index=2)

if st.button("Predict"):
    df = load_data(ticker, period)
    plot_stock(df)

    X, y, scaler = prepare_data(df)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model((X_train.shape[1], 1))
    predicted = train_and_predict(model, X_train, y_train, X_test, scaler)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    plot_prediction(df, actual, predicted, split + 60)
    st.success("✅ Prediction Completed!")

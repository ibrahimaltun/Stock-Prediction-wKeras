# Stock-Prediction-wKeras
Building an LSTM Model for Stock Price Forecasting with Keras

This repository contains a deep learning model built using Keras and TensorFlow to predict future stock prices based on historical data. The model leverages Long Short-Term Memory (LSTM) networks, which are well-suited for time series forecasting tasks such as stock market prediction.


# Overview
- The goal of this project is to:
- Preprocess historical stock data
- Train an LSTM model on closing prices
- Predict future stock movements
- Visualize actual vs predicted prices


# Model Highlights
- Architecture: Single or multi-layer LSTM
- Input Features: Historical closing prices
- Output: Next-day predicted closing price
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam



# Project Structure
```
├── models/                        # Saved model files
├── plots/                         # Visual outputs
├── MicrosoftStock.csv             # Raw and processed stock data
├── forecast_stock_prices.ipynb    # Python notebook for training and prediction
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```


# Installation
```
git clone https://github.com/ibrahimaltun/Stock-Prediction-wKeras.git
cd Stock-Prediction-wKeras
pip install -r requirements.txt
```

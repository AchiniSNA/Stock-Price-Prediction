import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MaxNLocator
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the trained model
model = load_model('D:\JN\Latest_stock_price_model.keras')

st.title('Stock Dashboard')


ticker_options = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
ticker = st.sidebar.selectbox('Select Ticker', ticker_options)
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

data = yf.download(ticker, start=start_date, end=end_date)

st.subheader('Stock Price Changes')

st.write(data)

# Create a new figure
fig = go.Figure()

# Add traces for each line
fig.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open Price'))
fig.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High Price'))
fig.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low Price'))
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Adj Close Price'))

# Update layout
fig.update_layout(title=ticker, xaxis_title='Date', yaxis_title='Price')

# Display the figure
st.plotly_chart(fig)


st.subheader('Volume')
fig6 = px.line(data, x= data.index, y= data['Volume'], title = ticker)
st.plotly_chart(fig6)




###################################################################

# Analysis for each column

# Prediction For Future
st.title('Pricing Analysis')

# Get the tab selection from the user
selected_tab = st.radio("Select Data", ['Open Price1','High Price1','Low Price1','Close Price1','Adj Close Price1','Stock Volume1'])





def plot_price_chart(data, price_type):
    st.subheader(f'{price_type} Price Line Chart')
    # Create a Plotly figure
    fig = go.Figure()
    # Add trace for the price
    fig.add_trace(go.Scatter(x=data.index, y=data[price_type], mode='lines', name=f'{price_type} Price'))
    # Update layout
    fig.update_layout(title=f'{price_type} Price Line Chart', xaxis_title='Date', yaxis_title=f'{price_type} Price')
    # Display the Plotly figure
    st.plotly_chart(fig)

    st.subheader('Price vs MA50 vs MA100 vs MA200')   
    # Calculate the 100-day and 200-day moving averages
    ma_50_days = data[price_type].rolling(50).mean()
    ma_100_days = data[price_type].rolling(100).mean()
    ma_200_days = data[price_type].rolling(200).mean()
    # Create a Plotly figure
    fig_ma = go.Figure()
    # Add trace for the 50-day moving average
    fig_ma.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='50-day MA', line=dict(color='yellow')))
    # Add trace for the 100-day moving average
    fig_ma.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='100-day MA', line=dict(color='red')))
    # Add trace for the 200-day moving average
    fig_ma.add_trace(go.Scatter(x=data.index, y=ma_200_days, mode='lines', name='200-day MA', line=dict(color='blue')))
    # Add trace for the Close prices
    fig_ma.add_trace(go.Scatter(x=data.index, y=data[price_type], mode='lines', name=price_type, line=dict(color='green')))
    # Update layout
    fig_ma.update_layout(title='50-day and 100-day and 200-day Moving Averages vs Price', xaxis_title='Date', yaxis_title='Price')
    # Display the Plotly figure
    st.plotly_chart(fig_ma)

#st.title('Stock Dashboard')
#selected_tab = st.sidebar.selectbox('Select Tab', ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Adj Close Price', 'Stock Volume'])

if selected_tab == 'Open Price1':
    plot_price_chart(data, 'Open')
elif selected_tab == 'High Price1':
    plot_price_chart(data, 'High')
elif selected_tab == 'Low Price1':
    plot_price_chart(data, 'Low')
elif selected_tab == 'Close Price1':
    plot_price_chart(data, 'Close')
elif selected_tab == 'Adj Close Price1':
    plot_price_chart(data, 'Adj Close')
elif selected_tab == 'Stock Volume1':
    plot_price_chart(data, 'Volume')

######################################################################################################################################

# Prediction For Future
st.sidebar.title('Predictions For Future')

# Get input from the user
days = st.sidebar.text_input('Number of Days need to Predict (0 - 30): ')

# Validate the input
try:
    days = int(days)
    if days < 0 or days > 30:
        st.sidebar.error("Please enter a number between 0 and 30.")
    else:
        st.sidebar.success(f"Prediction will be made for {days} days.")
except ValueError:
    st.sidebar.error("Please enter a valid number.")


# Get the tab selection from the user
selected_tab = st.sidebar.radio("Select Data", ['Open Price','High Price','Low Price','Close Price','Adj Close Price'])


def predict_and_plot(tab_name, column_name):
    st.write(f"Predicting {tab_name} data...")

    splitting_len = int(len(data) * 0.7)
    x_test = pd.DataFrame(data[column_name][splitting_len:])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[[column_name]])

    x_data = []
    y_data = []

    for i in range(20, len(scaled_data)):    ############changed
        x_data.append(scaled_data[i - 20:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    predictions = model.predict(x_data)

    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    ploting_data = pd.DataFrame(
        {
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_pre.reshape(-1)
        },
        index=data.index[splitting_len + 20:]                     #########changed
    )

    st.subheader(f'Original {tab_name} Price vs Predicted {tab_name} price')
    # Plot only the original test data and predicted test data using Plotly
    fig = go.Figure()

    # Add trace for original test data
    fig.add_trace(go.Scatter(x=ploting_data.index, y=ploting_data['original_test_data'], mode='lines',
                             name='original_test_data'))

    # Add trace for predicted test data
    fig.add_trace(go.Scatter(x=ploting_data.index, y=ploting_data['predictions'], mode='lines',
                             name='predictions'))

    # Update layout
    fig.update_layout(title=f'Original {tab_name} Price vs Predicted {tab_name} Price', xaxis_title='Date',
                      yaxis_title='Price')

    # Display the Plotly figure
    st.plotly_chart(fig)

    st.subheader("Original values vs Predicted values")
    st.write(ploting_data)

    # Calculate the error
    mae = mean_absolute_error(ploting_data['original_test_data'], ploting_data['predictions'])
    mse = mean_squared_error(ploting_data['original_test_data'], ploting_data['predictions'])
    rmse = np.sqrt(mse)

    # Display the error
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    # Assuming model is your trained model and data is your dataset
    # Get the most recent data
    recent_data = data.tail(100)  # Assuming you want to use the last 100 data points for prediction

    # Preprocess the data
    scaled_recent_data = scaler.fit_transform(recent_data[[column_name]])  # Assuming 'scaler' is your MinMaxScaler used during training

    # Generate input sequences
    input_sequence = scaled_recent_data[-20:]  # Assuming you use sequences of length 100 during training                ############changed
    input_sequence = np.reshape(input_sequence, (1, input_sequence.shape[0], input_sequence.shape[1]))  # Reshape for model input

    # Predict prices for the next 10 days
    future_predictions = []
    for _ in range(days):
        prediction = model.predict(input_sequence)
        future_predictions.append(prediction)
        input_sequence = np.append(input_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)  # Update input sequence with the new prediction

    # Postprocess the predictions
    inv_future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))  # Inverse transform to get actual price values

    # Create index for the next 10 days
    next_10_days_index = pd.date_range(start=recent_data.index[-1], periods=10, closed='right')

    # Plot the historical data and predicted prices
    fig = go.Figure()

    # Add trace for historical data
    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data[column_name], mode='lines', name='Historical Data'))

    # Add trace for predicted prices
    fig.add_trace(go.Scatter(x=next_10_days_index, y=inv_future_predictions.flatten(), mode='lines',
                             name='Predicted Prices'))

    # Update layout
    fig.update_layout(title=f'Historical {tab_name} Prices and Predicted Prices for Next {days} Days',
                      xaxis_title='Date', yaxis_title='Price')

    # Display the Plotly figure
    st.plotly_chart(fig)

    st.subheader(f'Predictions for Next {days} Days')
    st.write(inv_future_predictions)


if selected_tab == 'Open Price':
    predict_and_plot('Open Price', 'Open')
elif selected_tab == 'High Price':
    predict_and_plot('High Price', 'High')
elif selected_tab == 'Low Price':
    predict_and_plot('Low Price', 'Low')
elif selected_tab == 'Close Price':
    predict_and_plot('Close Price', 'Close')
elif selected_tab == 'Adj Close Price':
    predict_and_plot('Adj Close Price', 'Adj Close')

###############################################################

from stocknews import StockNews
#with news:
st.StockNews(f'News of {ticker}')
sn = StockNews(ticker, save_news=False)
df_news = sn.resd_rss()
for i in range(10):
    st.subheader(f'News {i+1}')
    st.write(df_news['published'][i])
    st.write(df_news['title'][i])
    st.write(df_news['summary'][i])
    title_sentiment = df_news['sentiment_title'][i]
    st.write(f'Title Sentiment {title_sentiment}')
    news_sentimnet = df_news['sentiment_summary'][i]
    st.write(f'News Sentiment {news_sentimnet}')
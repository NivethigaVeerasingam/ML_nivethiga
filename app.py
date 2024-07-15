from flask import Flask, render_template, redirect, request, url_for
import yfinance as yf
import pandas as pd
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import date
import itertools
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split, learning_curve


app = Flask(__name__)

users = {
    'admin': 'nevi123'
}

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

def crete_chart(data,ticker):
    fig = px.line(data, x=data.index, y='Close', title=f'{ticker} Closing Prices')
    
    # Convert plotly figure to JSON
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json



def get_best_arima_model(data, p_values, d_values, q_values):
    best_aic = np.inf
    best_order = None
    best_model = None

    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(data, order=(p, d, q))
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
                best_model = model_fit
        except:
            continue
    return best_order, best_model

def predict_stock_prices_arima(data, days):
    historical_data = data['Close'].dropna().asfreq('B').fillna(method='ffill')
    train_size = int(len(historical_data) * 0.8)
    train, test = historical_data[:train_size], historical_data[train_size:] 
    close_prices = data['Close']
    

    p_values = range(0, 6)
    d_values = range(0, 3)
    q_values = range(0, 6)

    best_order, best_model = get_best_arima_model(close_prices, p_values, d_values, q_values)
    forecast = best_model.forecast(steps=days)
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Close': forecast})
    forecast_df.set_index('Date', inplace=True)
    return forecast_df

def predict_stock_prices_LR(data, days):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values

    # Split data into train and test sets (80% train, 20% test)
    train_size = int(len(data) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Initialize and fit Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future prices
    predictions = model.predict(X)

    # Predict future prices
    future_dates = pd.date_range(start=data.index[-1], periods=days + 1, freq='B')[1:]  # Generate future dates
    future_X = np.arange(len(data), len(data) + days).reshape(-1, 1)
    future_predictions = model.predict(future_X)

    # Create DataFrame for predicted prices
    prediction_df = pd.DataFrame({'Date': data.index.tolist() + future_dates.tolist(), 'Close': np.concatenate([predictions, future_predictions])})
    prediction_df.set_index('Date', inplace=True)

    return prediction_df

  
def create_all_plot(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low'))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_plot(data, prediction_df, ticker,model):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Data'))
    fig.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Close'], mode='lines', name='Predicted Data', line=dict(color='red', dash='dot')))
    fig.update_layout(title=f'{ticker} Closing Prices and Predictions with {model}', xaxis_title='Date', yaxis_title='Close')
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json

def create_scatter_chart(data, prediction_df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='markers', name='Historical Data'))
    fig.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Close'], mode='lines', name='Predicted Data', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f'{ticker} Historical and Predicted Closing Prices', xaxis_title='Date', yaxis_title='Price')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        days = 7
        start_date = '2023-01-01'
    else:
        ticker = 'AAPL'
        days = 7
        start_date = '2023-01-01'
    
    end_date = date.today().strftime("%Y-%m-%d")
    data = fetch_stock_data(ticker, start_date, end_date)
    if not data.empty:
        open_price = format(data['Open'].iloc[-1], '.2f')
        high_price = format(data['High'].iloc[-1], '.2f')
        low_price = format(data['Low'].iloc[-1], '.2f')
        close_price = format(data['Close'].iloc[-1], '.2f')
    else:
        open_price = high_price = low_price = close_price = None
    
    chart_view=crete_chart(data,ticker)
    
    return render_template('index.html',graph_json=chart_view, ticker=ticker, days=days, open_price=open_price, high_price=high_price, low_price=low_price, close_price=close_price)
    # prediction_df = predict_stock_prices(data, int(days))
    # graph_json = create_plot(data, prediction_df, ticker)
    # return render_template('index.html',close=100, ticker=ticker, days=days,date=start_date)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
        
            return redirect(url_for('index'))
        else:
            
            return render_template('login.html', message='Invalid username or password')

    # For GET requests, render the login form
    return render_template('login.html')

@app.route('/history', methods=['GET', 'POST'])
def history():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
    else:
        ticker = 'AAPL'
        start_date = '2020-01-01'
        end_date = date.today().strftime("%Y-%m-%d")
    
    data = fetch_stock_data(ticker, start_date, end_date)

    chart_view= create_all_plot(data)

    return render_template('history.html',graph_json=chart_view,ticker=ticker,start_date=start_date,end_date=end_date,data=data.tail(10).to_html(classes='table table-striped'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        ticker = request.form['ticker']
        days = int(request.form['days'])
        start_date = request.form['start_date']
        model = request.form['model']  # Assuming ARIMA model is selected
    else:
        ticker = 'AAPL'
        days = 7
        model = 'LinearRegression'
        start_date = '2023-01-01'
    
    end_date = date.today().strftime("%Y-%m-%d")
    data = fetch_stock_data(ticker, start_date, end_date)
    
    if model == 'arima':
        prediction_df = predict_stock_prices_arima(data, days)
        graph_json = create_plot(data, prediction_df, ticker,model)
    elif model == 'LinearRegression':
        prediction_df = predict_stock_prices_LR(data, days)
        graph_json = create_plot(data, prediction_df, ticker,model)

    return render_template('predict.html', graph_json=graph_json, ticker=ticker, days=days, model=model, start_date=start_date)

@app.route('/help', methods=['GET', 'POST'])
def help():
    return render_template('help.html')






if __name__ == '__main__':
    app.run(debug=True)
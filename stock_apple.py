import yfinance as yf

def download_apple_stock_data():
    ticker_symbol = "AAPL"
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(period="max")
    file_path = "apple.csv"
    historical_data.to_csv(file_path)
    print("Stock data downloaded and saved successfully.")

download_apple_stock_data()

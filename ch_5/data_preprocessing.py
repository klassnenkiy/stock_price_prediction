import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def filter_data_by_tickers(data, tickers):
    return data[data['TICKER'].isin(tickers)]

def preprocess_data(data):
    data['TRADEDATE'] = pd.to_datetime(data['TRADEDATE'])
    
    return data

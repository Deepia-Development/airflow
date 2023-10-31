import logging
import requests 
from joblib import  Parallel, delayed
import pandas as pd 
import numpy as np 
import pandas.tseries.holiday as hol
import pandas_ta as ta
from datetime import datetime
from datetime import timedelta 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from dotenv import load_dotenv

# The setting for call the api
load_dotenv()
api_key = os.getenv("API_KEY")
headers = {'X-CoinAPI-Key': api_key}



class Extract:
    
    def __init__(self, time: int):
        self.time = time
        self.headers = headers
        self.initial_time = datetime.now() - timedelta(minutes = self.time) 
        self.final_time = datetime.now()
    
    def get_data(self):
        df_list = []  # Lista para almacenar DataFrames parciales
        start_date_str = self.initial_time.strftime("%Y-%m-%dT%H:%M:%S")
        end_date_str = self.final_time.strftime("%Y-%m-%dT%H:%M:%S")
        start_date = datetime.strptime(start_date_str, "%Y-%m-%dT%H:%M:%S")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%dT%H:%M:%S")
        delta = timedelta(minutes=101)  # Solicitar 100 valores por lote + 1 minuto para la próxima solicitud
        while start_date < end_date:
            next_date = start_date + delta
            next_date = min(next_date, end_date)  # Asegurarse de que no se exceda el final_time
            url = f"https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_BTC_USD/history?period_id=1MIN&time_start={start_date.isoformat()}&time_end={next_date.isoformat()}"
            res = requests.get(url, headers = self.headers)
            raw_data = res.json()
            if raw_data:
                df_list.append(pd.DataFrame(raw_data))
            start_date = next_date + timedelta(minutes=1)  # Avanzar 1 minuto para la próxima solicitud
        df = pd.concat(df_list, ignore_index=True)
        return df

    

class Transform(Extract):
    
    def __init__(self, time: int):
        super().__init__(time)
    
    def process_data(self):
        df = self.get_data()
        USFederalHolidayCalendar = hol.USFederalHolidayCalendar
        window_size = 20
        df['SMA_price_open'] = df['price_open'].rolling(window=window_size).mean()
        df['SMA_price_low'] = df['price_low'].rolling(window=window_size).mean()
        df['SMA_price_high'] = df['price_high'].rolling(window=window_size).mean()
        df['SMA_price_close'] = ta.sma(df['price_close'], length=14)
        df['SMA_volume_traded'] = ta.sma(df['volume_traded'], length=14)
        #RSI
        df['RSI_price_close'] = ta.rsi(df['price_close'], length=14)
        df['RSI_volume_traded'] = ta.rsi(df['volume_traded'], length=14)
        # MACD
        short_window=12
        long_window=26
        signal_window=9
        short_ema = df['price_close'].ewm(span=short_window, adjust=False).mean()
        long_ema = df['price_close'].ewm(span=long_window, adjust=False).mean()
        df['MACD'] = short_ema - long_ema
        # Stochastic Oscillator
        df['Stochastic_Oscillator'] = 100 * (df['price_close'] - df['price_low'].rolling(window=14).min()) / (df['price_high'].rolling(window=14).max() - df['price_low'].rolling(window=14).min())
        # Momentum
        df['Momentum'] = df['price_close'] - df['price_close'].shift(4)
        # Bollinger Bands
        df['Bollinger_Middle_Band'] = df['price_close'].rolling(window=20).mean()
        df['Bollinger_Upper_Band'] = df['Bollinger_Middle_Band'] + 2*df['price_close'].rolling(window=20).std()
        df['Bollinger_Lower_Band'] = df['Bollinger_Middle_Band'] - 2*df['price_close'].rolling(window=20).std()
        # ATR - Average True Range
        high_low = df['price_high'] - df['price_low']
        high_close = np.abs(df['price_high'] - df['price_close'].shift())
        low_close = np.abs(df['price_low'] - df['price_close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(window=14).mean()
        # OBV (On-Balance Volume)
        df['Daily_Return'] = df['price_close'].pct_change()
        df['Direction'] = np.where(df['Daily_Return'] >= 0, 1, -1)
        df.loc[0, 'Direction'] = 0
        df['OBV'] = (df['volume_traded'] * df['Direction']).cumsum()
        # Lagged Features
        df['Lagged_price_open'] = df['price_open'].shift(periods=1)
        df['Lagged_price_low'] = df['price_low'].shift(periods=1)
        df['Lagged_price_high'] = df['price_high'].shift(periods=1)
        df['Lagged_price_close'] = df['price_close'].shift(1)
        df['Lagged_volume_traded'] = df['volume_traded'].shift(1)
        # Change and Percentage Change
        df['Change_price_close'] = df['price_close'].diff()
        df['Change_volume_traded'] = df['volume_traded'].diff()
        df['Pct_Change_price_close'] = df['price_close'].pct_change()
        df['Pct_Change_volume_traded'] = df['volume_traded'].pct_change()
        # Rolling Metrics
        df['Rolling_Mean_price_close'] = df['price_close'].rolling(window=20).mean()
        df['Rolling_Std_price_close'] = df['price_close'].rolling(window=20).std()
        # Time-Based Features
        df['Minute'] = pd.to_datetime(df['time_period_start']).dt.minute
        df['Hour'] = pd.to_datetime(df['time_period_start']).dt.hour
        df['DayOfWeek'] = pd.to_datetime(df['time_period_start']).dt.dayofweek
        # Holiday and Weekend Indicators
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=pd.to_datetime(df['time_period_start']).min(), end=pd.to_datetime(df['time_period_start']).max())
        df['Holiday'] = pd.to_datetime(df['time_period_start']).isin(holidays)
        df['Weekend'] = df['DayOfWeek'].isin([5, 6])
        df = df.replace([np.inf, -np.inf], np.nan)
        trash = ['time_period_start', 'time_period_end', 'time_open','time_close']
        mean = df.drop(columns = trash)
        # replace infinite values with NaN
        df_all = df.fillna(mean.mean())  # fill NaN values with column mean
        return df_all
    
    def process_data1(self):
        df = self.get_data()
        USFederalHolidayCalendar = hol.USFederalHolidayCalendar
        window_size = 20
        df['SMA_price_open'] = df['price_open'].rolling(window=window_size).mean()
        df['SMA_price_low'] = df['price_low'].rolling(window=window_size).mean()
        df['SMA_price_high'] = df['price_high'].rolling(window=window_size).mean()
        df['SMA_price_close'] = df['price_close'].rolling(window=14).mean()
        df['SMA_volume_traded'] = df['volume_traded'].rolling(window=14).mean()
        # RSI - Relative Strength Index
        delta = df['price_close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI_price_close'] = 100 - (100 / (1 + rs))
        # MACD
        short_window=12
        long_window=26
        signal_window=9
        short_ema = df['price_close'].ewm(span=short_window, adjust=False).mean()
        long_ema = df['price_close'].ewm(span=long_window, adjust=False).mean()
        df['MACD'] = short_ema - long_ema
        # ADX
        plus_dm = df['price_high'].diff()
        minus_dm = df['price_low'].diff()
        tr = df['price_high'] - df['price_low']
        tr = tr.where(tr > 0, 0)
        tr14 = tr.rolling(window=14).sum()
        plus_dm14 = plus_dm.rolling(window=14).sum()
        minus_dm14 = minus_dm.rolling(window=14).sum()
        plus_di14 = 100 * plus_dm14 / tr14
        minus_di14 = 100 * minus_dm14 / tr14
        dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
        df['ADX'] = dx.rolling(window=14).mean()
        # ROC
        df['ROC'] = df['price_close'].pct_change(periods=10)
        # MFI
        typical_price = (df['price_high'] + df['price_low'] + df['price_close']) / 3
        money_flow = typical_price * df['volume_traded']
        positive_money_flow = money_flow.where(df['price_close'] > df['price_close'].shift(1), 0).rolling(window=14).sum()
        negative_money_flow = money_flow.where(df['price_close'] < df['price_close'].shift(1), 0).rolling(window=14).sum()
        money_ratio = positive_money_flow / negative_money_flow
        df['MFI'] = 100 - (100 / (1 + money_ratio))
        # WAD
        tr_high = np.maximum(df['price_high'], df['price_close'].shift(1))
        tr_low = np.minimum(df['price_low'], df['price_close'].shift(1))
        tr_range = tr_high - tr_low
        df['WAD'] = np.where(df['price_close'] > df['price_close'].shift(1), df['price_close'] - tr_low, df['price_close'] - tr_high)
        # 50-day and 200-day moving averages
        df['MA_50'] = df['price_close'].rolling(window=50).mean()
        df['MA_200'] = df['price_close'].rolling(window=200, min_periods = 1).mean()
        # Stochastic Oscillator
        df['Stochastic_Oscillator'] = 100 * (df['price_close'] - df['price_low'].rolling(window=14).min()) / (df['price_high'].rolling(window=14).max() - df['price_low'].rolling(window=14).min())
        # Momentum
        df['Momentum'] = df['price_close'] - df['price_close'].shift(4)
        # Bollinger Bands
        df['Bollinger_Middle_Band'] = df['price_close'].rolling(window=20).mean()
        df['Bollinger_Upper_Band'] = df['Bollinger_Middle_Band'] + 2*df['price_close'].rolling(window=20).std()
        df['Bollinger_Lower_Band'] = df['Bollinger_Middle_Band'] - 2*df['price_close'].rolling(window=20).std()
        # ATR - Average True Range
        high_low = df['price_high'] - df['price_low']
        high_close = np.abs(df['price_high'] - df['price_close'].shift())
        low_close = np.abs(df['price_low'] - df['price_close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(window=14).mean()
        # OBV (On-Balance Volume)
        df['Daily_Return'] = df['price_close'].pct_change()
        df['Direction'] = np.where(df['Daily_Return'] >= 0, 1, -1)
        df.loc[0,'Direction'] = 0
        df['OBV'] = (df['volume_traded'] * df['Direction']).cumsum()
        # Lagged Features
        df['Lagged_price_open'] = df['price_open'].shift(periods=1)
        df['Lagged_price_low'] = df['price_low'].shift(periods=1)
        df['Lagged_price_high'] = df['price_high'].shift(periods=1)
        df['Lagged_price_close'] = df['price_close'].shift(1)
        df['Lagged_volume_traded'] = df['volume_traded'].shift(1)
        # Change and Percentage Change
        df['Change_price_close'] = df['price_close'].diff()
        df['Change_volume_traded'] = df['volume_traded'].diff()
        df['Pct_Change_price_close'] = df['price_close'].pct_change()
        df['Pct_Change_volume_traded'] = df['volume_traded'].pct_change()
        # Rolling Metrics
        df['Rolling_Mean_price_close'] = df['price_close'].rolling(window=20).mean()
        df['Rolling_Std_price_close'] = df['price_close'].rolling(window=20).std()
        # Time-Based Features
        df['Minute'] = pd.to_datetime(df['time_period_start']).dt.minute
        df['Hour'] = pd.to_datetime(df['time_period_start']).dt.hour
        df['DayOfWeek'] = pd.to_datetime(df['time_period_start']).dt.dayofweek
        # Holiday and Weekend Indicators
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=pd.to_datetime(df['time_period_start']).min(), end=pd.to_datetime(df['time_period_start']).max())
        df['Holiday'] = pd.to_datetime(df['time_period_start']).isin(holidays)
        df['Weekend'] = df['DayOfWeek'].isin([5, 6])
        df = df.replace([np.inf, -np.inf], np.nan)  # replace infinite values with NaN
        df = df.drop(columns=['time_period_start', 'time_period_end', 'time_open', 'time_close'])
        df = df.fillna(df.mean())  # fill NaN values with column mean
        return df


    def preprocess_for_future(self):
        df = self.process_data1()
        look_back = 30 # 1
        # FEATURES = ['SMA_price_close','RSI_price_close','Lagged_price_open','Change_price_close','Pct_Change_price_close','Rolling_Mean_price_close','Rolling_Std_price_close','Minute','Hour','DayOfWeek']
        FEATURES = ['SMA_price_close','RSI_price_close','Momentum','MACD','Stochastic_Oscillator','OBV','ATR','price_open','SMA_volume_traded','ADX','ROC','MFI','WAD','MA_50','MA_200']
        TARGET = 'price_close'
        X = df[FEATURES].values
        y = df[TARGET].values
        # Normalize the features
        y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Transform the data into a format suitable for LSTM
        X_arr, y_arr = [], []
        for i in range(len(X)-look_back-1):
            a = X[i:(i+look_back), :]
            X_arr.append(a)
            y_arr.append(y_scaled[i + look_back])
        X, y_scaled = np.array(X_arr), np.array(y_arr)
        # X = np.array(X_arr, dtype=np.float32)  # Change data type to float32
        # y_scaled = np.array(y_arr, dtype=np.float32)
        # Reshape input to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], look_back, X.shape[2]))
        return X, y_scaler,df 
    

# raw_data = Transform(time = 1)
# s = raw_data.process_data1()
# print(s)

# raw_data = Extract(time = 1)
# data = raw_data.get_data()
# data = data[['time_period_start','price_close']]
# print(data)
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/NinoNinov/Deep-Learning-Project-SoftUni/blob/main/deep_learning_project_msft.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from datetime import datetime
from transformers import pipeline
from dateutil.relativedelta import relativedelta
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
import feedparser  # Import the missing module
from transformers import pipeline  # Import Hugging Face pipeline
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from statsmodels.tsa.seasonal import STL
from tensorflow.keras import layers, Model, Sequential
from dateutil.relativedelta import relativedelta

# %% [markdown]
# # Deep Learning for Stock Price Prediction with LSTM and Transformers
#
# ## Project Overview  
# This project focuses on building regression models using **LSTM (Long Short-Term Memory)** and **Transformers** to predict stock prices. The models are trained on **two datasets**:  
# - A **2-year stock dataset**  
# - A **35-year stock dataset**  
#
# The goal is to **forecast stock prices for the next 15 days** and compare the actual vs. predicted values. The primary objective is to evaluate and compare the performance of **LSTM and Transformer models**, demonstrating that Transformers provide better results due to their more reliable architecture.  
#
# ---
#
# ## Data Preprocessing  
#
# 1. **Data Collection & Feature Engineering**  
#    - The dataset consists of **11 features** extracted from stock data.  
#
# 2. **Outlier Removal**  
#    - Outliers are removed based on [method to be specified].  
#
# 3. **Data Splitting**  
#    - The dataset is divided into **training, validation, and test sets**.  
#    - The split ratios differ for **LSTM and Transformer models** due to their unique requirements.  
#
# 4. **Normalization**  
#    - The data is normalized using the **Z-score** method.  
#    - A **stride of 1** is applied to improve model generalization.  
#
# 5. **Reshaping for Model Input**  
#    - The dataset is divided into **X (features) and y (target values)**.  
#    - The feature matrix **X** is reshaped from **2D to 3D** to meet the input requirements of the models.  
#
# ---
#
# ## Model Development  
#
# ### 1. **LSTM Model**  
#    - Designed and implemented a **regression LSTM architecture** for stock price prediction.  
#
# ### 2. **Transformer Model**  
#    - Developed a **Transformer-based regression model**, leveraging self-attention mechanisms.  
#
# ### 3. **Model Compilation & Training**  
#    - Both models are compiled with appropriate loss functions and optimizers.  
#    - The models are trained on the prepared dataset.  
#
# ### 4. **Prediction & Forecasting**  
#    - The models predict stock prices using two forecasting techniques:  
#      - **Recursive forecast (one-step prediction at a time)**  
#      - **Recursive forecast with multiple steps ahead**  
#
# ---
#
# ## Model Evaluation  
#
# 1. **Testing on Unseen Data**  
#    - The trained models are tested on actual stock data they haven't seen before.  
#
# 2. **Comparison of Predicted vs. Actual Values**  
#    - The forecasted values from both models are compared with real stock prices.  
#
# 3. **Performance Analysis**  
#    - Evaluation metrics are used to determine the accuracy and reliability of **LSTM vs. Transformer** models.  
#    - The findings help in understanding why Transformers may outperform LSTMs in long-term stock price predictions.  
#
# ---
#
# ## Conclusion  
# This project aims to demonstrate that **Transformers provide more accurate predictions** than LSTMs due to their superior architecture, better handling of long-term dependencies, and ability to capture trends more effectively in stock data.  
#
# ---
#
# ## Future Work  
# - Experiment with different hyperparameters and architectures.  
# - Extend the forecasting period beyond 15 days.  
# - Incorporate external market indicators for better prediction accuracy.  
# - Optimize Transformer models for improved efficiency.  
#
# ---
#
# ### Technologies Used  
# - **Python**  
# - **TensorFlow/Keras**  
# - **Pandas, NumPy, Scikit-learn**  
# - **Matplotlib, Seaborn** (for visualization)  
#

# %% [markdown]
# <span style="font-size:35px;"><strong>Structure of the Project</strong></span>
#
# <span style="font-size:17px; font-weight:bold;">&nbsp;&nbsp;1. Prepare Data</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;1.1 General Input</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;1.2 LSTM Input</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;1.3 Transformers Input</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;1.4 Fetch Data</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;1.5 Extract Working Days and Actual Values</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;1.6 Feature Engineering</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;1.7 STL Decomposition</span>
#
# <span style="font-size:17px; font-weight:bold;">&nbsp;&nbsp;2. Split (Train, Validation, and Test), Normalize, Stride, Reshape of Data</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;2.1 Split the Data to Train, Validation, and Test</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;2.2 Normalization of Data</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;2.3 Apply Stride and Divide Dataset to X and Y</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;2.4 Reshape 2D to 3D (Needed for X)</span>
#
# <span style="font-size:17px; font-weight:bold;">&nbsp;&nbsp;3. LSTM Model</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;3.1 Architecture of Model</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;3.2 Compile the Model</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;3.3 Fit the Model</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;3.4 Predict with the Model</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;3.5 Recursive Forecast (One Step Forecast)</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;3.6 Multi-Step Forecasting</span>
#
# <span style="font-size:17px; font-weight:bold;">&nbsp;&nbsp;4. Time-Series Transformer</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;4.1 Model Architecture</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;4.2 Compile the Model</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;4.3 Fit the Model</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;4.4 Predict with the Model</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;&nbsp;&nbsp;4.5 Multi-Step Forecasting</span>
#
# <span style="font-size:17px;">&nbsp;&nbsp;5. Sentiment Analysis of Stock</span>
#
# <span style="font-size:17px; font-weight:bold;">&nbsp;&nbsp;6. Summary</span>
#
#
#
#

# %% [markdown]
# <div style="background-color: #e8d21d; color: #000000; font-size: 18px; padding: 5px;"> 1. Prepare Data
# </div>

# %% [markdown]
# <div style="background-color: #e8d21d; color: #000000; font-size: 18px; padding: 5px;"> 1.1 General Input
# </div>

# %%
instrument = 'MSFT'  # Stock symbol
end_date = '2024-12-30'  # End date for historical data
keyword = "Microsoft"  # Keyword for specific context or visualization (optional)
stride = 1  # Step size for moving window of time series
feature_regression = ['SMA_20', 'lag_close_2', 'lag_close_1', 'EMA_20', 'RSI']  # List of feature columns
features_classification = ["lag_return_1", "SMA_20", "RSI", "Spread_HL", "day_of_week"]
target_regression = 'Close'  # The target column to predict
target_classification = 'return_type'  # The target column to predict
va_window = 100 #Period for creating of Moving Averages

# %% [markdown]
# <div style="background-color: #e8d21d; color: #000000; font-size: 18px; padding: 5px;"> 1.2 LSTM Input
# </div>

# %%
lstm_period = 2 #Set years for LSTM
start_date_lstm = datetime.strptime(end_date, '%Y-%m-%d') #start date of transformer
start_date_lstm = start_date_lstm - relativedelta(years=lstm_period) #end date of transformer

time_steps_lstm = 60  # Time steps for LSTM input
batch_size_lstm = 64  # Number of samples per gradient update
epochs_lstm = 30  # Number of training epochs

learning_rate_lstm = 0.001  # Learning rate for the Adam optimizer
train_percentage_lstm = 0.7  # Percentage of data for training
val_percentage_lstm = 0.15  # Percentage of data for validation
test_percentage_lstm = 0.15  # Percentage of data for testing

# %% [markdown]
# <div style="background-color: #e8d21d; color: #000000; font-size: 18px; padding: 5px;"> 1.3 Transformers Input
# </div>

# %%
transformer_period = 35 #Set years for transformer
start_date_transformer = datetime.strptime(end_date, '%Y-%m-%d') #start date of transformer
start_date_transformer = start_date_transformer - relativedelta(years=transformer_period) #end date of transformer

time_steps_transformer = 60  # Time steps for LSTM input
batch_size_transformer = 32  # Number of samples per gradient update
epochs_transfprmer = 30  # Number of training epochs
learning_rate_transformer = 0.001 # Learning rate for the Adam optimizer

train_percentage_transformer = 0.9  # Percentage of data for training
val_percentage_transformer = 0.05  # Percentage of data for validation
test_percentage_transformer = 0.05  # Percentage of data for testing


# %% [markdown]
# <div style="background-color: #e8d21d; color: #000000; font-size: 18px; padding: 5px;"> 1.4 Fetch Data
# </div>

# %% id="b8zgdUV14d6t" outputId="6a675517-9926-4469-dbc6-4df6a0970ed4"
def get_historical_data(instrument, start_date, end_date):
    dat = yf.Ticker(instrument)
    data = dat.history(start=start_date, end=end_date)
    data = data.reset_index()
    
    return data

data_lstm = get_historical_data(instrument, start_date_lstm, end_date)
data_transformer = get_historical_data(instrument, start_date_transformer, end_date)

# %% [markdown]
# <div style="background-color: #e8d21d; color: #000000; font-size: 18px; padding: 5px;"> 1.5 Extract Working days and Actual Values
# </div>

# %% [markdown]
# 1.5.1 Extract Working Days

# %%
import datetime


def get_working_days(year):
    # Define weekends (Saturday and Sunday)
    weekends = {5, 6}  # 5 = Saturday, 6 = Sunday
    
    # Define public holidays (customize based on your country)
    public_holidays = [
        datetime.date(year, 1, 1),   # New Year's Day
        datetime.date(year, 7, 4),  # Independence Day
        datetime.date(year, 12, 25) # Christmas Day
    ]
    
    # Add floating holidays (e.g., Martin Luther King Jr. Day - 3rd Monday of January)
    # MLK Jr. Day (3rd Monday of January)
    jan_1_weekday = datetime.date(year, 1, 1).weekday()
    mlk_day = datetime.date(year, 1, 1) + datetime.timedelta(days=(14 + (7 - jan_1_weekday) % 7))
    public_holidays.append(mlk_day)
    
    # Thanksgiving (4th Thursday of November)
    nov_1_weekday = datetime.date(year, 11, 1).weekday()
    thanksgiving = datetime.date(year, 11, 1) + datetime.timedelta(days=(3 - nov_1_weekday + 7 * 3) % 7)
    public_holidays.append(thanksgiving)
    
    # Generate all days in the year
    days = np.arange(datetime.date(year, 1, 1), datetime.date(year + 1, 1, 1), dtype="datetime64[D]")
    
    working_days = [
        day.astype(datetime.date) for day in days
        if day.astype(datetime.date).weekday() not in weekends and day.astype(datetime.date) not in public_holidays
    ]
    
    return working_days

year = 2025
working_days = get_working_days(year)
working_days = pd.DataFrame(working_days, columns=['Date'])
working_days = working_days.head(60)


# %% [markdown]
# 1.5.2 Extract Actual values

# %%
def get_close_prices(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)     
    close_prices = data[['Close']].reset_index()      
    close_prices.columns = ['Date', 'Close']
    return close_prices
    
close_prices = get_close_prices(ticker=instrument, start_date="2025-01-01", end_date="2025-12-20")
close_prices = close_prices[['Close']]
close_prices = close_prices.rename(columns={'Close': 'Actual'})


# %% [markdown] id="lgonXGJdoT43"
# <div style="background-color: #e8d21d; color: #000000; font-size: 18px; padding: 5px;"> 1.6 Feature Enginering
# </div>

# %% [markdown]
# Added Features: 1. SMA_20	2. EMA_20	3. RSI	3. day_of_week	4. lag_close_1	5. lag_close_2	6. return	7. lag_return_1	8. lag_return_2	9. Spread_OC	10. Spread_HL	11. return_type
#

# %%
def preprocess_data(data, va_window):
   
    def compute_rsi(series, period=14):
        """Calculate the Relative Strength Index (RSI) for a given series."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # Adding technical indicators
    data['SMA_20'] = data['Close'].rolling(window=va_window).mean()  # 20-day SMA
    data['EMA_20'] = data['Close'].ewm(span=va_window, adjust=False).mean()  # 20-day EMA
    data['RSI'] = compute_rsi(data['Close'])  # Relative Strength Index

    # Adding time-based features (day of the week)
    data.index = pd.to_datetime(data.index)
    data['day_of_week'] = data.index.dayofweek  # Day of the week (0=Monday, 6=Sunday)
    
    # Adding lagged features
    data['lag_close_1'] = data['Close'].shift(1)  # Previous day's closing price
    data['lag_close_2'] = data['Close'].shift(2)  # Two days ago closing price

    # Adding return rates (daily percentage change)
    data['return'] = data['Close'].pct_change()  # Daily return
    data['lag_return_1'] = data['return'].shift(1)  # Previous day's return
    data['lag_return_2'] = data['return'].shift(2)  # Two days ago return

    # Adding spread features
    data['Spread_OC'] = (data['Close'] - data['Open']) / data['Open']  # Open-Close spread
    data['Spread_HL'] = (data['High'] - data['Low']) / data['High']  # High-Low spread

    # Drop rows with NaN values (due to rolling operations and lagging)
    data.dropna(inplace=True)

    # Reset index and ensure 'Date' is in datetime format
    data.index = pd.to_datetime(data.index).date
    data.reset_index(drop=True, inplace=True)
    
    # Ensure 'Date' column is in datetime format and extract date portion
    data['Date'] = pd.to_datetime(data['Date']).dt.date

    # Round numeric columns to 2 decimal places
    data = data.round(2)

    # Create 'Volume_M' column and drop unnecessary columns
    data['Volume_M'] = data['Volume'] / 1000000
    data = data.drop(['Dividends', 'Stock Splits', 'Volume'], axis=1)

    # Adding return type (0 for positive returns, 1 for negative returns)
    data['return_type'] = data['return'].apply(lambda x: 0 if x > 0 else 1)

    # Reorder columns to place 'Volume_M' after 'Close'
    cols = list(data.columns)
    cols.insert(cols.index('Close') + 1, cols.pop(cols.index('Volume_M')))
    data = data[cols]

    # Return the cleaned and processed DataFrame
    return data


data_lstm = preprocess_data(data_lstm, va_window)
data_transformer = preprocess_data(data_transformer, va_window)


# %% [markdown] id="zT1plBOFrvIN"
# <div style="background-color: #e8d21d; color: #000000; font-size: 18px; padding: 5px;"> 1.7 STL Decomposition and Outlier Removal
#
# </div>
#

# %% id="bNLfVJdKrvIN" outputId="8dbe46b0-b29f-4f20-c17a-6b67486c6813"
def preprocess_stock_data(data, seasonal_period=5):
    
    
    # Step 1: Convert 'Date' to datetime and set as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Step 2: Ensure regular frequency (Business days for stock data)
    data = data.asfreq('B').ffill()  # Forward-fill missing values

    # Step 3: Fill missing values in 'Close' column
    data['Close'] = data['Close'].interpolate(method='linear').ffill().bfill()

    # Step 4: Apply STL decomposition on the 'Close' series
    stl = STL(data['Close'], seasonal=seasonal_period, robust=True)
    result = stl.fit()

    # Step 5: Residual component and outlier detection
    residual = result.resid
    threshold = 3 * np.std(residual)
    outliers = np.abs(residual) > threshold

    # Step 6: Replace outliers with interpolated values
    data.loc[outliers, 'Close'] = np.nan
    data['Close'] = data['Close'].interpolate(method='linear').ffill().bfill()

    # Step 7: Reset index for final output
    data.reset_index(inplace=True)
    
    return data


data_lstm = preprocess_stock_data(data_lstm)
data_transformer = preprocess_stock_data(data_transformer)

# %%
data_lstm

# %%
data_transformer


# %% [markdown]
# <div style="background-color: #ff6e40; color: #000000; font-size: 18px; padding: 5px;"> 2. Split (Train, Validation and Test), Normalise, Stride, Reshape of Data
# </div>

# %% [markdown] id="DS1XNt5HbmTr"
# <div style="background-color: #ff6e40; color: #000000; font-size: 18px; padding: 5px;"> 2.1 Split the data to Train, Validation and Split
# </div>

# %%
def split_data(data, time_steps, train_percentage, val_percentage):
  
    # Calculate split indices
    train_end = int(len(data) * train_percentage)
    val_end = train_end + int(len(data) * val_percentage)

    # Split the data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Adjust date columns based on time_steps
    train_date = train_data['Date'][time_steps:]
    val_date = val_data['Date'][time_steps:]
    test_date = test_data['Date'][time_steps:]

    return train_data, val_data, test_data, train_date, val_date, test_date

train_data_lstm, val_data_lstm, test_data_lstm, train_date_lstm, val_date_lstm, test_date_lstm = split_data(data_lstm, time_steps_lstm, train_percentage_lstm, val_percentage_lstm)
train_data_transformer, val_data_transformer, test_data_transformer, train_date_transformer, val_date_transformer, test_date_transformer = split_data(data_transformer, time_steps_transformer, train_percentage_transformer, val_percentage_transformer)

# %% [markdown]
# Plot Train, Validation and Split - Close

# %% [markdown]
# LSTM - 2 Years
#

# %% id="ona79XnervIP" outputId="7635377a-edec-49c4-a636-37e6fa5f9e16"
plt.figure(figsize=(16,8))

plt.plot(train_data_lstm['Date'], train_data_lstm['Close'])
plt.plot(val_data_lstm['Date'], val_data_lstm['Close'])
plt.plot(test_data_lstm['Date'], test_data_lstm['Close'])

plt.legend(['Train', 'Validation', 'Test'])
plt.show()

# %% [markdown]
# Transformers - 35 Years

# %%
plt.figure(figsize=(16,8))

plt.plot(train_data_transformer['Date'], train_data_transformer['Close'])
plt.plot(val_data_transformer['Date'], val_data_transformer['Close'])
plt.plot(test_data_transformer['Date'], test_data_transformer['Close'])

plt.legend(['Train', 'Validation', 'Test'])
plt.show()


# %% [markdown] id="y94271a1rvIP"
# <div style="background-color: #ff6e40 ; color: #000000; font-size: 18px; padding: 5px;">2.2 Normalisation of Data
# </div>

# %% [markdown]
# 2.2.1 Normalisation of whole DataFrame for Model Processing

# %% id="I7tlmCNarvIQ"
def normalize_dataframe(df: pd.DataFrame, drop_columns: list = None) -> pd.DataFrame:
    if drop_columns:
        df = df.drop(columns=drop_columns)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(df.values)
    normalized_data = normalizer(df.values)
    normalized_df = pd.DataFrame(normalized_data.numpy(), columns=df.columns)

    return normalized_df, normalizer

train_data_normalised_lstm, normalizer_train_lstm = normalize_dataframe(train_data_lstm, drop_columns=['Date'])
val_data_normalised_lstm, normalizer_val_lstm = normalize_dataframe(val_data_lstm, drop_columns=['Date'])
test_data_normalised_lstm, normalizer_test_lstm = normalize_dataframe(test_data_lstm, drop_columns=['Date'])

train_data_normalised_transformer, normalizer_train_transformer = normalize_dataframe(train_data_transformer, drop_columns=['Date'])
val_data_normalised_transformer, normalizer_val_transformer = normalize_dataframe(val_data_transformer, drop_columns=['Date'])
test_data_normalised_transformer, normalizer_test_transformer = normalize_dataframe(test_data_transformer, drop_columns=['Date'])

# %% [markdown]
# 2.2.2 Independent Normalisation of Column Close

# %%
close_column_lstm = train_data_lstm[["Close"]]
close_column_transformer = train_data_transformer[["Close"]]


def normalize_dataframe(df: pd.DataFrame, drop_columns: list = None) -> pd.DataFrame:
    if drop_columns:
        df = df.drop(columns=drop_columns)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(df.values)
    normalized_data = normalizer(df.values)
    normalized_df = pd.DataFrame(normalized_data.numpy(), columns=df.columns)

    return normalized_df, normalizer

train_data_close_lstm, normalizer_close_lstm= normalize_dataframe(close_column_lstm)
train_data_close_transformer, normalizer_close_transformer= normalize_dataframe(close_column_transformer)


# %% [markdown]
# <div style="background-color: #ff6e40; color: #000000; font-size: 18px; padding: 5px;"> 2.3 Appy Stride and Divide Dataset to X and Y
# </div>

# %% [markdown] id="Ko1VwZbQrvIQ"
# Train Stride

# %% id="wgIjfQ3prvIQ" outputId="53108483-00e9-4d98-dc76-b97ffa2a9f76"
def create_sequences_with_stride_2d(data, time_steps, stride, feature_columns, target_column):

    X, Y = [], []

    # Loop through the data with stride and create sequences
    for i in range(0, len(data) - time_steps, stride):
        sequence_features = data.iloc[i:i + time_steps][feature_columns].values  # Get the feature data for the current sequence
        X.append(sequence_features.flatten())  # Flatten into 1D array
        Y.append(data.iloc[i + time_steps][target_column]) # Append the target (next time-step's target column)

    return np.array(X), np.array(Y)

#For Regression
X_train_lstm, y_train_lstm = create_sequences_with_stride_2d(train_data_normalised_lstm, time_steps_lstm, stride, feature_regression, target_regression)
X_train_transformer, y_train_transformer = create_sequences_with_stride_2d(train_data_normalised_transformer, time_steps_transformer, stride, feature_regression, target_regression)

print("X shape LSTM:", X_train_lstm.shape)
print("Y shape LSTM:", y_train_lstm.shape)
print("X shape Transformer:", X_train_transformer.shape)
print("Y shape Transformer:", y_train_transformer.shape)

# %% [markdown] id="oJMsSJekrvIQ"
# Validation Stride

# %% id="1efWWCTO5tSc" outputId="233b0646-c794-4392-d19f-94657e7f8d57"
X_val_lstm, y_val_lstm = create_sequences_with_stride_2d(val_data_normalised_lstm, time_steps_lstm, stride, feature_regression, target_regression)
X_val_transformer, y_val_transformer = create_sequences_with_stride_2d(val_data_normalised_transformer, time_steps_transformer, stride, feature_regression, target_regression)

print("X shape LSTM:", X_val_lstm.shape)
print("Y shape LSTM:", y_val_lstm.shape)
print("X shape Transformer:", X_val_transformer.shape)
print("Y shape Transformer:", y_val_transformer.shape)

# %% [markdown] id="EspcYq3SrvIR"
# Test Stride

# %% id="XkwEz-xirvIR" outputId="da3fb82d-9136-4050-fc22-416befb55b2b"
X_test_lstm, y_test_lstm = create_sequences_with_stride_2d(test_data_normalised_lstm, time_steps_lstm, stride, feature_regression, target_regression)
X_test_transformer, y_test_transformer = create_sequences_with_stride_2d(test_data_normalised_transformer, time_steps_transformer, stride, feature_regression, target_regression)

print("X shape LSTM:", X_test_lstm.shape)
print("Y shape LSTM:", y_test_lstm.shape)
print("X shape Transformer:", X_test_transformer.shape)
print("Y shape Transformer:", y_test_transformer.shape)


# %% [markdown]
# <div style="background-color: #ff6e40; color: #000000; font-size: 18px; padding: 5px;"> 2.4 Reshape 2D to 3D (need for X)
# </div>

# %% [markdown] id="3UhY3h0irvIR"
# Train Reshape

# %% id="19rvCzfprvIS" outputId="3888867d-09a3-4948-9fdd-e17deaa5123f"
def convert_2d_to_3d(X, time_steps, num_features):

    # Reshape the 2D data into 3D
    X_3d = X.reshape((X.shape[0], time_steps, num_features))
    return X_3d

num_features_regression = len(feature_regression)  
num_features_classification = len(features_classification)  

X_train_lstm = convert_2d_to_3d(X_train_lstm, time_steps_lstm, num_features_regression)
X_train_transformer = convert_2d_to_3d(X_train_transformer, time_steps_transformer, num_features_regression)

print("X_3d shape LSTM:", X_train_lstm.shape)
print("X_3d shape Transformer:", X_train_transformer.shape)

# %% [markdown] id="ibmMO9pDrvIS"
# Validation Reshape

# %% id="gygVrBpurvIS" outputId="749224a0-c73a-4fec-90a8-9c72d6f4eaff"
X_val_lstm = convert_2d_to_3d(X_val_lstm, time_steps_lstm, num_features_regression)
X_val_transformer = convert_2d_to_3d(X_val_transformer, time_steps_transformer, num_features_regression)

print("X_3d shape LSTM:", X_val_lstm.shape)
print("X_3d shape Transformer:", X_val_transformer.shape)

# %% [markdown] id="Gfz2bjiErvIS"
# Test Reshape

# %% id="lSOzLeijrvIS" outputId="2307c614-c8dc-4394-933b-722b66ef5ab7"
X_test_lstm = convert_2d_to_3d(X_test_lstm, time_steps_lstm, num_features_regression)
X_test_transformer = convert_2d_to_3d(X_test_transformer, time_steps_transformer, num_features_regression)

print("X_3d shape LSTM:", X_test_lstm.shape)
print("X_3d shape Transformer:", X_test_transformer.shape)

# %% [markdown]
# <div style="background-color: #ddc3a5; color: #000000; font-size: 18px; padding: 5px;"> 3. LSTM Model

# %% [markdown] id="Dp1ZYpghrvIS"
# <div style="background-color: #ddc3a5; color: #000000; font-size: 18px; padding: 5px;"> 3.1 Architecture of Model

# %% [markdown]
# #### LSTM Model Architecture Overview  
#
# This **LSTM-based architecture** processes **60 time steps with 5 features**, capturing temporal dependencies in stock data.  
#
# #### **Key Components:**  
# - **LSTM Layer (64 units):** Extracts sequential patterns from input data.  
# - **Dense Layers (32 units each):** Further feature transformation.  
# - **Dropout Layers:** Prevent overfitting by randomly disabling neurons during training.  
# - **Multi-Output Design:**  
#   - `y_output` (1 unit) → Predicts the next stock price.  
#   - `x_output` (5 units) → Auxiliary predictions, possibly reconstructing input features.  
#
# This structure balances **feature extraction, regularization, and multi-task learning**, enhancing stock price forecasting accuracy. 🚀  
#

# %% id="JTJgNBPLact3"
inputs = layers.Input(shape=(time_steps_lstm, 5))

# LSTM layer with L2 regularization and dropout
x = layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, 
                kernel_regularizer=regularizers.l2(0.01))(inputs)

# Dense layers with L2 regularization and dropout (without Batch Normalization)
x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Dropout(0.2)(x)

# Output layers
y_output = layers.Dense(1, name="y_output")(x)  # Predicts y (1 value)
x_output = layers.Dense(5, name="x_output")(x)  # Predicts x (5 values)

# Create the model
final_model_regression = Model(inputs=inputs, outputs=[y_output, x_output])

# Print model summary
final_model_regression.summary()

# %% [markdown] id="mnvl99UMrvIU"
# <div style="background-color: #ddc3a5; color:rgb(0, 0, 0); font-size: 18px; padding: 5px;"> 3.2 Compile the Model
# </div>

# %% [markdown]
# The **LSTM-based regression model** is compiled using the **Mean Squared Error (MSE) loss function** and the **Adam optimizer** with a specified learning rate.  
#
# ### **Key Components:**  
# - **Loss Function:**  
#   - `"mse"` (Mean Squared Error) is used for both `y_output` (stock price prediction) and `x_output` (auxiliary task).  
#   - MSE minimizes the squared differences between actual and predicted values, making it suitable for regression tasks.  
#
# - **Optimizer:**  
#   - **Adam (Adaptive Moment Estimation):**  
#     - Efficient gradient-based optimization.  
#     - Controlled by `learning_rate_lstm`, ensuring appropriate step sizes during training.  
#
# - **Evaluation Metrics:**  
#   - `"mean_absolute_error"` (MAE) is used to measure prediction accuracy for both outputs.  
#   - MAE provides an intuitive measure of error by averaging the absolute differences between actual and predicted values.  
#
# This configuration ensures **stable training, accurate stock price forecasting, and auxiliary feature learning**. 🚀  
#

# %% id="NLPAbce5am9_"
final_model_regression.compile(
    loss={"y_output": "mse", "x_output": "mse"},
    optimizer=Adam(learning_rate=learning_rate_lstm),
    metrics={"y_output": "mean_absolute_error", "x_output": "mean_absolute_error"}
)

# %% [markdown] id="e3jMDWoWrvIU"
# <div style="background-color: #ddc3a5; color: #000000; font-size: 18px; padding: 5px;"> 3.3 Fit the Model
# </div>

# %% [markdown]
# The **LSTM-based regression model** is trained using stock data, optimizing for both **stock price prediction (`y_output`)** and **auxiliary feature learning (`x_output`)**.  
#
# ### **Key Training Steps:**  
# 1. **Fitting the Model:**  
#    - The model is trained on **X_train_lstm** with two outputs:  
#      - `y_output`: The actual stock price (`y_train_lstm`).  
#      - `x_output`: The last time step of the input sequence (`X_train_lstm[:, -1, :]`).  
#
# 2. **Training Configuration:**  
#    - **Epochs:** Controlled by `epochs_lstm`.  
#    - **Batch Size:** Set by `batch_size_lstm`.  
#    - **Validation Data:**  
#      - Uses `X_val_lstm` with corresponding `y_output` and `x_output` for model validation.  
#    - **Verbose Output:** Displays real-time training progress (`verbose=1`).  
#
# 3. **Loss Tracking:**  
#    - **Training Loss:**  
#      - `train_loss_y`: Loss for `y_output` (stock price prediction).  
#      - `train_loss_x`: Loss for `x_output` (auxiliary task).  
#    - **Validation Loss:**  
#      - `val_loss_y`: Validation loss for `y_output`.  
#      - `val_loss_x`: Validation loss for `x_output`.  
#
# 4. **Epoch Tracking:**  
#    - `epochs_range`: Stores the epoch range for potential visualization (e.g., loss plots).  
#
# This setup ensures **effective training and validation**, allowing the model to **learn stock price patterns while preserving feature representations**.
#

# %% id="Lu4O09lurvIU" outputId="da597830-61cf-4046-f86b-9aac17fb7a12"
history = final_model_regression.fit(
    X_train_lstm, {"y_output": y_train_lstm, "x_output": X_train_lstm[:, -1, :]},  # y and last timestep of x
    epochs=epochs_lstm,
    batch_size=batch_size_lstm,
    validation_data=(X_val_lstm, {"y_output": y_val_lstm, "x_output": X_val_lstm[:, -1, :]}),
    verbose=1
)

train_loss_y = history.history['y_output_loss']
val_loss_y = history.history['val_y_output_loss']
train_loss_x = history.history['x_output_loss']
val_loss_x = history.history['val_x_output_loss']

epochs_range = range(1, len(train_loss_y) + 1)

# %%
plt.figure(figsize=(16, 6))

# Plot loss for y_output
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss_y, label='Training Loss (y)', marker='o')
plt.plot(epochs_range, val_loss_y, label='Validation Loss (y)', marker='o')
plt.title('Training and Validation Loss for y_output')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Plot loss for x_output
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss_x, label='Training Loss (x)', marker='o')
plt.plot(epochs_range, val_loss_x, label='Validation Loss (x)', marker='o')
plt.title('Training and Validation Loss for x_output')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# %% [markdown]
# <div style="background-color: #ddc3a5; color: #000000; font-size: 18px; padding: 5px;"> 3.4 Predict with the Model
# </div>

# %%
train_predictions, _ = final_model_regression.predict(X_train_lstm, verbose=0)  # Ignore x_output

plt.figure(figsize=(16, 8))

# Use train_date_lstm for the x-axis
plt.plot(train_date_lstm, train_predictions, label='Train Predictions')
plt.plot(train_date_lstm, y_train_lstm, label='Actual Values')

plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Training Predictions vs Actual Values')
plt.legend()

plt.xticks(rotation=45)  # Rotate dates for better readability

plt.show()


# %% [markdown]
# MSE and RMSE for Training Predictions

# %%
mse = tf.keras.losses.MeanSquaredError()
mean_squared_error_value = mse(y_train_lstm, train_predictions)

# Calculate the Root Mean Squared Error (RMSE)
rmse = tf.sqrt(mean_squared_error_value)
print("MSE:", mean_squared_error_value.numpy())
print("RMSE:", rmse.numpy())

# %% [markdown] id="ci4Qx0gdrvIe"
# Validation Predictions

# %%
val_predictions, _ = final_model_regression.predict(X_val_lstm, verbose=0) # Ignore x_output

plt.figure(figsize=(16,8))
plt.plot(val_date_lstm, val_predictions, label='Validation Predictions')
plt.plot(val_date_lstm, y_val_lstm, label='Actual Values')

# Add labels, legend, and title
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Validation Predictions vs Actual Values')
plt.legend()

# Rotate date labels for better readability
plt.xticks(rotation=45)

plt.show()

# %% [markdown]
# MSE and RMSE for Validation Predictions

# %%
mse = tf.keras.losses.MeanSquaredError()
mean_squared_error_value = mse(y_val_lstm, val_predictions)

rmse = tf.sqrt(mean_squared_error_value)
print("MSE:", mean_squared_error_value.numpy())
print("RMSE:", rmse.numpy())

# %% [markdown] id="Uc0_R98mrvIf"
# Test Data Predictions

# %%
test_data_predictions, _ = final_model_regression.predict(X_test_lstm, verbose=0) # Ignore x_output

plt.figure(figsize=(16,8))
plt.plot(test_date_lstm, test_data_predictions, label='Test Predictions')
plt.plot(test_date_lstm, y_test_lstm, label='Actual Values')

# Add labels, legend, and title
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Test Predictions vs Actual Values')
plt.legend()

# Rotate date labels for better readability
plt.xticks(rotation=45)

plt.show()

# %% [markdown]
# MSE and RMSE for Test

# %%
mse = tf.keras.losses.MeanSquaredError()
mean_squared_error_value = mse(y_test_lstm, test_data_predictions)

rmse = tf.sqrt(mean_squared_error_value)
print("MSE:", mean_squared_error_value.numpy())
print("RMSE:", rmse.numpy())

# %% [markdown]
#
# <div style="background-color: #ddc3a5; color: #000000; font-size: 18px; padding: 5px;"> 3.5 Recursive Forecast(One Step Forecst)
# </div>

# %% [markdown]
# Code below performs a 15-day recursive stock price forecast using a trained LSTM-based regression model.
#
# 1. **Initialize Input**:
#     - The sequence starts with the last sequence from the test dataset: `X_test_lstm[-1]`.
#     - This initial input is reshaped to match the model's input dimensions: `(1, time_steps_lstm, 5)`.
#
# 2. **Iterative Forecasting**:
#     - A loop runs for **15 days**, predicting the next stock price (`y_output`) and the auxiliary features (`x_output`) at each step.
#
# 3. **Prediction and Input Update**:
#     - **After each prediction**:
#         - The predicted stock price (`y_output`) is stored.
#         - The input sequence is shifted left using `np.roll`, removing the oldest time step.
#         - The predicted features (`x_output`) are inserted at the last time step to simulate real-world sequential forecasting.
#
# Process Overview:
#
# - The model generates predictions iteratively, using its own past outputs as inputs for future steps.
#

# %%
last_sequence = X_test_lstm[-1]  
current_sequence = last_sequence.copy()  
predictions = []

current_sequence = current_sequence.reshape((1, time_steps_lstm, 5))

for day in range(15):
    prediction_y, prediction_x = final_model_regression.predict(current_sequence, verbose=0)  # Predict both y and x in one step

    predictions.append(prediction_y[0, 0])  # Store y prediction

    current_sequence = np.roll(current_sequence, shift=-1, axis=1)  #shift left
    current_sequence[0, -1, :] = prediction_x  # Insert new prediction for x at last time step


# %% [markdown]
# Denormalisation

# %%
mean = normalizer_close_lstm.mean.numpy()
variance = normalizer_close_lstm.variance.numpy()
std_dev = tf.sqrt(variance).numpy()

denormalized_lstm = predictions * std_dev + mean
denormalized_lstm = pd.DataFrame(denormalized_lstm).T
denormalized_lstm = denormalized_lstm.rename(columns={0: 'Forecast'})

# %% [markdown]
# Concatinate Actual with Forecasted

# %%
one_step_lstm = pd.concat([working_days, denormalized_lstm, close_prices], axis=1)

# %% [markdown]
# Plot Actual vs Forecast (used second axis because model did not perfect first predicted day very accurately)

# %%
fig, ax1 = plt.subplots(figsize=(14, 7))

# Primary Y-Axis (Actual)
ax1.plot(one_step_lstm['Date'][:15], one_step_lstm['Actual'][:15], label='Actual', color='red', linestyle='--', marker='x', alpha=0.7)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Actual Price', fontsize=12, color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Secondary Y-Axis (Forecast)
ax2 = ax1.twinx()
ax2.plot(one_step_lstm['Date'], one_step_lstm['Forecast'], label='Forecast', color='blue', linestyle='-', marker='o', alpha=0.7)
ax2.set_ylabel('Forecast Price', fontsize=12, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

fig.suptitle('Actual vs Forecast Prices', fontsize=16)

# Set x-axis ticks and labels
ax1.set_xticks(one_step_lstm['Date'][:15])  # Use appropriate range for the x-ticks
ax1.set_xticklabels(one_step_lstm['Date'][:15], rotation=45)

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

ax1.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# %% [markdown]
#
# <div style="background-color: #ddc3a5; color: #000000; font-size: 18px; padding: 5px;"> 3.6 Multi-Step Foreasting
# </div>

# %% [markdown]
# This code implements **multi-step stock price prediction** using a trained LSTM model. It predicts the stock prices for the next 90 days by breaking the prediction into smaller steps (e.g., `n_steps` at a time). 
#
# ### How it works:
# 1. **Initialization**:
#    - The process begins by using the **last sequence** from the test dataset (`X_test_lstm[-1]`).
#    - This initial sequence is reshaped to match the model’s input dimensions.
#
# 2. **Prediction Loop**:
#    - The loop runs to predict for a total of 90 days, predicting in chunks of `n_steps` at each iteration.
#    - For each iteration, the model predicts both the stock prices (`prediction_y`) and the auxiliary features (`prediction_x`).
#
# 3. **Updating Input**:
#    - After each prediction:
#      - The predicted stock prices (`prediction_y`) are stored in the `predictions` list.
#      - The input sequence is **shifted left** by `n_steps` to make room for the new predictions.
#      - The predicted features (`prediction_x`) are added at the end of the sequence to continue the iterative forecasting process.
#
# ### Key Benefits:
# - This approach allows the model to generate predictions iteratively, using its own previous outputs as inputs for future steps.
# - By predicting in smaller chunks and updating the sequence step-by-step, the model simulates a more realistic forecasting scenario.
#
#

# %%
n_steps = 5  # Define how many steps to predict at once

last_sequence = X_test_lstm[-1]  
current_sequence = last_sequence.copy()  
predictions = []

current_sequence = current_sequence.reshape((1, time_steps_lstm, 5))  # Ensure correct shape

for step in range(75 // n_steps): # Adjust loop for multi-step prediction -  computes how many iterations of n(5) steps you can make to reach 90 steps.
    prediction_y, prediction_x = final_model_regression.predict(current_sequence, verbose=0)  # Predict multiple steps

    predictions.extend(prediction_y[0, :])  # Store all y predictions

    # Shift left by n_steps
    current_sequence = np.roll(current_sequence, shift=-n_steps, axis=1)  
    current_sequence[0, -n_steps:, :] = prediction_x  # Insert new predictions at the end

# %% [markdown]
# Denormalisation

# %%
mean = normalizer_close_lstm.mean.numpy()
variance = normalizer_close_lstm.variance.numpy()
std_dev = tf.sqrt(variance).numpy()

denormalized_multi_step = predictions * std_dev + mean
denormalized_multi_step = pd.DataFrame(denormalized_multi_step).T
denormalized_multi_step = denormalized_multi_step.rename(columns={0: 'Forecast'})

# %% [markdown]
# Concatinate Actual with Forecast

# %%
mult_step_lstm = pd.concat([working_days, denormalized_multi_step, close_prices], axis=1)

# %% [markdown]
# Plot Actual vs Forecast (used second axis because model did not perfect first predicted day very accurately)

# %%
fig, ax1 = plt.subplots(figsize=(14, 7))

# Primary Y-Axis (Actual)
ax1.plot(mult_step_lstm['Date'][:15], mult_step_lstm['Actual'][:15], label='Actual', color='red', linestyle='--', marker='x', alpha=0.7)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Actual Price', fontsize=12, color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Secondary Y-Axis (Forecast)
ax2 = ax1.twinx()
ax2.plot(mult_step_lstm['Date'], mult_step_lstm['Forecast'], label='Forecast', color='blue', linestyle='-', marker='o', alpha=0.7)
ax2.set_ylabel('Forecast Price', fontsize=12, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

fig.suptitle('Actual vs Forecast Prices', fontsize=16)

# Set x-axis ticks and labels (ensure correct number of ticks)
ax1.set_xticks(mult_step_lstm['Date'][:15])  # Use the first 16 data points for the x-axis ticks
ax1.set_xticklabels(mult_step_lstm['Date'][:15], rotation=45)

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

ax1.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# %% [markdown]
# <div style="background-color: #77c593; color: #000000; font-size: 18px; padding: 5px;">4. Time-Series Transformer
# </div>

# %% [markdown]
# This code defines a **transformer-based neural network** using TensorFlow's Keras API for a sequence modeling task.
#
# ### **1. Input Layer**
# - The input shape is **(60, 5)**, meaning the model processes **sequences of length 60** with **5 features per timestep**.
#
# ### **2. Positional Encoding**
# - A positional embedding layer is used to **add position information** to the input sequences.
# - The embeddings are expanded to match the batch dimension and then **added** to the input.
#
# ### **3. Transformer Blocks (3 Layers)**
# - **Multi-Head Self-Attention** (4 heads, key dimension = 32).
# - **Dropout** (0.1 probability) for regularization.
# - **Layer Normalization** to stabilize training.
# - **Feedforward Dense Layer** (64 units, ReLU activation).
# - **Dropout** and **Layer Normalization** repeated.
#
# ### **4. Feature Pooling**
# - **Global Average Pooling** reduces the sequence dimension to a fixed-size vector.
#
# ### **5. Shared Dense Layers**
# - **Dense Layer (32 units, ReLU activation)**.
# - **Dense Layer (32 units, ReLU activation)**.
#
# ### **6. Dual Output Predictions**
# - **y_output**: A single value prediction (`Dense(1)`).
# - **x_output**: A vector prediction of size **5** (`Dense(5)`).
#
# ### **7. Model Creation**
# - The model is built using the **Keras Functional API** with **one input** and **two outputs**.
#
#
#
#

# %% [markdown]
# <div style="background-color: #77c593; color: #000000; font-size: 18px; padding: 5px;">4.1 Model Architecture
#

# %%
inputs = layers.Input(shape=(60, 5))

# Positional Encoding
position_embeddings = layers.Embedding(input_dim=60, output_dim=5)(tf.range(start=0, limit=60, delta=1)) 
position_embeddings = tf.expand_dims(position_embeddings, axis=0)  # Add batch dimension
position_encoded_input = layers.Add()([inputs, position_embeddings])

# First Transformer Block
transformer_block_1 = layers.MultiHeadAttention(num_heads=4, key_dim=32)(position_encoded_input, position_encoded_input)
transformer_block_1 = layers.Dropout(0.1)(transformer_block_1)
transformer_block_1 = layers.LayerNormalization()(transformer_block_1)
transformer_block_1 = layers.Dense(64, activation='relu')(transformer_block_1)
transformer_block_1 = layers.Dropout(0.1)(transformer_block_1)
transformer_block_1 = layers.LayerNormalization()(transformer_block_1)

# Second Transformer Block
transformer_block_2 = layers.MultiHeadAttention(num_heads=4, key_dim=32)(transformer_block_1, transformer_block_1)
transformer_block_2 = layers.Dropout(0.1)(transformer_block_2)
transformer_block_2 = layers.LayerNormalization()(transformer_block_2)
transformer_block_2 = layers.Dense(64, activation='relu')(transformer_block_2)
transformer_block_2 = layers.Dropout(0.1)(transformer_block_2)
transformer_block_2 = layers.LayerNormalization()(transformer_block_2)

# Third Transformer Block
transformer_block_3 = layers.MultiHeadAttention(num_heads=4, key_dim=32)(transformer_block_2, transformer_block_2)
transformer_block_3 = layers.Dropout(0.1)(transformer_block_3)
transformer_block_3 = layers.LayerNormalization()(transformer_block_3)
transformer_block_3 = layers.Dense(64, activation='relu')(transformer_block_3)
transformer_block_3 = layers.Dropout(0.1)(transformer_block_3)
transformer_block_3 = layers.LayerNormalization()(transformer_block_3)

# Global Average Pooling
pooled_output = layers.GlobalAveragePooling1D()(transformer_block_3)

# Shared Dense Layers (as in the LSTM model)
x = layers.Dense(32, activation='relu')(pooled_output)
x = layers.Dense(32, activation='relu')(x)

y_output = layers.Dense(1, name="y_output")(x)  # Predicts y (1 value)
x_output = layers.Dense(5, name="x_output")(x)  # Predicts x (5 values)

model_transformer = Model(inputs=inputs, outputs=[y_output, x_output])  # Create a Keras functional API model with 2 outputs

# %% [markdown]
# <div style="background-color: #77c593; color: #000000; font-size: 18px; padding: 5px;">4.2 Compile the model Model
#

# %% [markdown]
# ### **1. Model Creation**
# - The model is built using the **Keras Functional API** with:
#   - **One input** (`inputs`).
#   - **Two outputs** (`y_output` and `x_output`).
#
# ### **2. Model Compilation**
# - **Loss Function**:  
#   - Both outputs (`y_output` and `x_output`) use **Mean Squared Error (MSE)**.
# - **Optimizer**:  
#   - Adam optimizer with a **custom learning rate (`learning_rate_transformer`)**.
# - **Evaluation Metrics**:  
#   - Both outputs use **Mean Absolute Error (MAE)**.
#
# ### **3. Model Summary**
# - `model_transformer.summary()` prints:
#   - **Layer structure**.
#   - **Number of parameters**.
#   - **Shape of inputs/outputs**.
#   - **Trainable parameters count**.
#

# %%
model_transformer = Model(inputs=inputs, outputs=[y_output, x_output])

model_transformer.compile(
    loss={"y_output": "mse", "x_output": "mse"},
    optimizer=Adam(learning_rate=learning_rate_transformer),
    metrics={"y_output": "mean_absolute_error", "x_output": "mean_absolute_error"}
)

model_transformer.summary()

# %% [markdown]
# <div style="background-color: #77c593; color: #000000; font-size: 18px; padding: 5px;">4.3 Fit the model
#

# %% [markdown]
# ### **1. Model Training**
# - The model is trained using `model_transformer.fit()`, with:
#   - **Input**: `X_train_transformer` (training data for the Transformer model).
#   - **Two Outputs**:
#     - `"y_output"`: `y_train_transformer` (target for `y_output`).
#     - `"x_output"`: The **last timestep** of `X_train_transformer` (assumed as the target for `x_output`).
#
# ### **2. Training Parameters**
# - **Epochs**: Set using `epochs_transformer`, ensuring it's an integer.
# - **Batch Size**: Defined by `batch_size_transformer`, also converted to an integer if necessary.
# - **Validation Data**:
#   - Uses `X_val_transformer` as input.
#   - `"y_output"`: `y_val_transformer`.
#   - `"x_output"`: The last timestep of `X_val_transformer`.
#
# ### **3. Training History**
# - The training process stores the **loss and metrics** in `history.history`.

# %%
history = model_transformer.fit(
    X_train_transformer,  # Input for the Transformer model
    {
        "y_output": y_train_transformer,
        "x_output": X_train_transformer[:, -1, :]  # Assuming last timestep for x_output
    },
    epochs=int(epochs_transfprmer),  # Force conversion to integer if needed
    batch_size=int(batch_size_transformer),  # Force conversion to integer if needed
    validation_data=(
        X_val_transformer,
        {
            "y_output": y_val_transformer,
            "x_output": X_val_transformer[:, -1, :]
        }
    ),
    verbose=1
)

history_dict = history.history
df_history = pd.DataFrame(history_dict)
df_history.to_csv("training_history_transformer.csv", index=False)

# %%

# Get the loss data for both outputs
train_loss_y = history.history['y_output_loss']
val_loss_y = history.history['val_y_output_loss']
train_loss_x = history.history['x_output_loss']
val_loss_x = history.history['val_x_output_loss']

# Create an epoch range for plotting
epochs_range = range(len(train_loss_y))  # Length of the loss data

plt.figure(figsize=(16, 6))

# Plot loss for y_output
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss_y, label='Training Loss (y)', marker='o')
plt.plot(epochs_range, val_loss_y, label='Validation Loss (y)', marker='o')
plt.title('Training and Validation Loss for y_output')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Plot loss for x_output
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss_x, label='Training Loss (x)', marker='o')
plt.plot(epochs_range, val_loss_x, label='Validation Loss (x)', marker='o')
plt.title('Training and Validation Loss for x_output')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# %% [markdown]
# <div style="background-color: #77c593; color: #000000; font-size: 18px; padding: 5px;">4.4 Predict with the model
#

# %% [markdown]
# Train Predictions

# %%
# Get predictions (returns both outputs)
train_predictions = model_transformer.predict(X_train_transformer)

# Extract y predictions (assuming it's the first output)
y_train_pred = train_predictions[0]  # Shape: (num_samples, 1)

plt.figure(figsize=(16, 8))

plt.plot(train_date_transformer, y_train_pred.squeeze(), label='Train Predictions')  
plt.plot(train_date_transformer, y_train_transformer.squeeze(), label='Actual Values')  

plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.title('Train Predictions vs Actual Values')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# MSE and RMSE for train

# %%
mse = tf.keras.losses.MeanSquaredError()
mean_squared_error_value = mse(
    y_true=y_train_transformer,  
    y_pred=y_train_pred         
)

rmse = tf.sqrt(mean_squared_error_value)
print("MSE:", mean_squared_error_value.numpy())
print("RMSE:", rmse.numpy())

# %% [markdown]
# Validation Predictions

# %%
# Get predictions (returns both outputs)
val_predictions = model_transformer.predict(X_val_transformer)

#Extract y predictions (assuming it's the first output)
y_val_pred = val_predictions[0]  # Shape: (num_samples, 1)

plt.figure(figsize=(16, 8))

# Plot predictions and actual values
plt.plot(val_date_transformer, y_val_pred.squeeze(), label='Validation Predictions')  # Squeeze to remove singleton dimension
plt.plot(val_date_transformer, y_val_transformer.squeeze(), label='Actual Values')  # Ensure same shape

# Add labels, legend, and title
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.title('Validation Predictions vs Actual Values')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# MSE and RMSE for predict

# %%
mse = tf.keras.losses.MeanSquaredError()
mean_squared_error_value = mse(
    y_true=y_val_transformer,  
    y_pred=y_val_pred          
)

rmse = tf.sqrt(mean_squared_error_value)
print("MSE:", mean_squared_error_value.numpy())
print("RMSE:", rmse.numpy())

# %% [markdown]
# Test Predictions

# %%
# Get predictions (returns both outputs)
test_predictions = model_transformer.predict(X_test_transformer)

#Extract y predictions (assuming it's the first output)
y_test_pred = test_predictions[0]  # Shape: (num_samples, 1)

plt.figure(figsize=(16, 8))

# Plot predictions and actual values
plt.plot(test_date_transformer, y_test_pred.squeeze(), label='Test Predictions')  # Squeeze to remove singleton dimension
plt.plot(test_date_transformer, y_test_transformer.squeeze(), label='Actual Values')  # Ensure same shape

plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.title('Test Predictions vs Actual Values')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# MSE and RMSE Train

# %%
mse = tf.keras.losses.MeanSquaredError()
mean_squared_error_value = mse(
    y_true=y_test_transformer,  # True y values
    y_pred=y_test_pred        # Predicted y values
)

rmse = tf.sqrt(mean_squared_error_value)
print("MSE:", mean_squared_error_value.numpy())
print("RMSE:", rmse.numpy())

# %% [markdown]
# <div style="background-color: #77c593; color: #000000; font-size: 18px; padding: 5px;">4.5 Recursive Forecast(One Step Forecst)
#

# %%
last_sequence_transformer = X_test_transformer[-1]  
current_sequence = last_sequence_transformer.copy()  
predictions_transformer_single = []

current_sequence = current_sequence.reshape((1, time_steps_transformer, 5))

for day in range(15):
    prediction_y, prediction_x = model_transformer.predict(current_sequence, verbose=0)  # Predict both y and x in one step
    predictions_transformer_single.append(prediction_y[0, 0])  # Store y prediction

    current_sequence = np.roll(current_sequence, shift=-1, axis=1)  #shift left
    current_sequence[0, -1, :] = prediction_x  # Insert new prediction for x at last time step


# %% [markdown]
# Denormalise

# %%
mean = normalizer_close_transformer.mean.numpy()
variance = normalizer_close_transformer.variance.numpy()
std_dev = tf.sqrt(variance).numpy()

denormalized_transformer_single = predictions_transformer_single * std_dev + mean
denormalized_transformer_single = pd.DataFrame(denormalized_transformer_single).T
denormalized_transformer_single = denormalized_transformer_single.rename(columns={0: 'Forecast'})

# %% [markdown]
# Concatinate

# %%
concatenated_df = pd.concat([working_days, denormalized_transformer_single, close_prices], axis=1)

# %% [markdown]
# Plot Actual vs Forecast (used second axis because model did not perfect first predicted day very accurately)

# %%
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot the actual values on the primary y-axis
ax1.plot(concatenated_df['Date'][:15], concatenated_df['Actual'][:15], 
         label='Actual', color='red', linestyle='--', marker='x', alpha=0.7)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Actual Prices', fontsize=12, color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.grid(alpha=0.3)

# Create a secondary y-axis for the forecast values
ax2 = ax1.twinx()
ax2.plot(concatenated_df['Date'], concatenated_df['Forecast'], 
         label='Forecast', color='blue', linestyle='-', marker='o', alpha=0.7)
ax2.set_ylabel('Forecast Prices', fontsize=12, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Actual vs Forecast Prices', fontsize=16)
plt.xticks(rotation=45)

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# %% [markdown]
# <div style="background-color: #77c593; color: #000000; font-size: 18px; padding: 5px;">4.5 Multi-Step Foreasting

# %%
n_steps = 5  # Define how many steps to predict at once

last_sequence = X_test_lstm[-1]  
current_sequence = last_sequence.copy()  
predictions_transformers_multi = []

current_sequence = current_sequence.reshape((1, time_steps_transformer, 5))  # Ensure correct shape

for step in range(75 // n_steps):  # Adjust loop for multi-step prediction -  computes how many iterations of n(5) steps you can make to reach 90 steps.
    prediction_y, prediction_x = model_transformer.predict(current_sequence, verbose=0)  # Predict multiple steps

    predictions_transformers_multi.extend(prediction_y[0, :])  # Store all y predictions

    # Shift left by n_steps
    current_sequence = np.roll(current_sequence, shift=-n_steps, axis=1)  
    current_sequence[0, -n_steps:, :] = prediction_x  # Insert new predictions at the end

# %% [markdown]
# Denormalisation

# %%
mean = normalizer_close_transformer.mean.numpy()
variance = normalizer_close_transformer.variance.numpy()
std_dev = tf.sqrt(variance).numpy()

denormalized_transformer_multi = predictions_transformers_multi * std_dev + mean
denormalized_transformer_multi = pd.DataFrame(denormalized_transformer_multi).T
denormalized_transformer_multi = denormalized_transformer_multi.rename(columns={0: 'Forecast'})

# %%
concatenated_df = pd.concat([working_days, denormalized_transformer_multi, close_prices], axis=1)

# %% [markdown]
# Plot Actual vs Forecast (used second axis because model did not perfect first predicted day very accurately)

# %%
plt.figure(figsize=(14, 7))

ax1 = plt.gca()
ax1.plot(concatenated_df['Date'][:15], concatenated_df['Actual'][:15], label='Actual', color='red', linestyle='--', marker='x', alpha=0.7)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price (Actual)', fontsize=12, color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Create second axis for Forecast values
ax2 = ax1.twinx() 
ax2.plot(concatenated_df['Date'], concatenated_df['Forecast'], label='Forecast', color='blue', linestyle='-', marker='o', alpha=0.7)
ax2.set_ylabel('Price (Forecast)', fontsize=12, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Title and formatting
plt.title('Actual vs Forecast Prices', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper left')

# Grid and show
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown] id="k3-ORTwVrvIg"
# <div style="background-color: #f3ca20; color: #000000; font-size: 18px; padding: 5px;"> 5. Sentiment Analysis of Stock
# </div>

# %%
ticker = instrument  # Example ticker symbol
keyword = keyword  # Example keyword to filter articles

# Initialize the Hugging Face pipeline with the FinBERT model
pipe = pipeline("text-classification", model="ProsusAI/finbert")

# Yahoo Finance RSS feed URL for the specified ticker
rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'

# Parse the RSS feed
feed = feedparser.parse(rss_url)

# Initialize variables to compute overall sentiment
total_score = 0
num_articles = 0

# Iterate through the RSS feed entries
for entry in feed.entries:
    # Check if the keyword is present in the article summary (case-insensitive)
    if keyword.lower() not in entry.summary.lower():
        continue

    # Print article details
    print(f'Title: {entry.title}')
    print(f'Link: {entry.link}')
    print(f'Published: {entry.published}')
    print(f'Summary: {entry.summary}')
    print('-' * 40)

    # Perform sentiment analysis on the article summary
    sentiment = pipe(entry.summary)[0]  # Only take the first result (most likely sentiment)

    # Print sentiment analysis results
    print(f'Sentiment: {sentiment["label"]}, Score: {sentiment["score"]}')
    print('-' * 40)

    # Adjust total score based on sentiment label
    if sentiment["label"].lower() == 'positive':
        total_score += sentiment["score"]
        num_articles += 1
    elif sentiment["label"].lower() == 'negative':
        total_score -= sentiment["score"]
        num_articles += 1

# Calculate and display the overall sentiment
if num_articles > 0:  # Avoid division by zero
    final_score = total_score / num_articles
    overall_sentiment = (
        "Positive" if final_score >= 0.15 else
        "Negative" if final_score <= -0.15 else
        "Neutral"
    )
    print(f'Overall Sentiment: {overall_sentiment} (Score: {final_score:.2f})')
else:
    print("No relevant articles found to compute sentiment.")


# %% [markdown]
# <div style="background-color:rgb(243, 116, 32); color: #000000; font-size: 18px; padding: 5px;"> 6. Summary
# </div>

# %% [markdown]
#
# In my work, I followed a **5-step approach** to analyze and forecast MSFT stock data, which included the following stages: **data fetching**, **feature engineering**, **preprocessing**, **modeling**, **evaluation**, and **forecasting**.
#
# - **Fetching Data**: I used different time periods for data retrieval:
#   - **LSTM model**: 2 years of historical data.
#   - **Transformer model**: 35 years of historical data.
#   
#   This choice was based on the distinct nature of the models, with Transformers benefiting from a larger time window to capture long-term dependencies.
#
# - **Feature Engineering and Preprocessing**: To improve model performance, I incorporated **11 additional features** alongside the basic stock data. Before preprocessing, I performed several important steps:
#   - **Outlier Removal**: I eliminated extreme values to ensure the data was clean and accurate.
#   - **Data Splitting**: I divided the data into **training**, **validation**, and **test sets** for proper model evaluation.
#   - **Normalization**: I normalized the data to ensure consistent scaling and facilitate model convergence.
#   - **Stride Application**: I applied a **stride of 1** to maintain the temporal sequence of the data.
#   - **Reshaping**: I reshaped the data from **2D to 3D** to meet the input requirements of both the LSTM and Transformer models.
#
# - **Modeling and Evaluation**: Both models achieved favorable results in terms of **Mean Squared Error (MSE)** on unseen test data, indicating good predictive performance. However, I noticed that the models performed weakly when extrapolating beyond the training data range. Additionally, the **denormalization process** didn’t work as expected, leading to challenges in transforming the forecasted results back to the original scale.
#
# - **Plotting Extrapolated Forecasts**: Due to the issues with denormalization, I decided to use the **second axis** for plotting the extrapolated forecasts. This helped to better visualize the behavior of the model when predicting future values outside the training window.
#
#

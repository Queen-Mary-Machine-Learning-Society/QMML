import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
import logging

def get_data(start_date: str, end_date: str) -> pd.DataFrame:

    gold = yf.download("GC=F", start=start_date, end=end_date, progress=False)
    silver = yf.download("SI=F", start=start_date, end=end_date, progress=False)

    data = pd.concat([gold['Close'],silver['Close']],axis=1)
    data.columns = ["Gold_Close", "Silver_Close"]

    data = data.dropna()

    return data

def get_linear_regression(data):

    corr = data["Gold_Close"].corr(data["Silver_Close"])

    slope, intercept, r_value, p_value, std_error = stats.linregress(
        data["Gold_Close"], data["Silver_Close"]
    )

    return corr, slope, intercept, r_value, p_value, std_error

def get_predictions(data, slope, intercept):

    result = data.copy() # Avoid SettingWithCopyWarning

    result['Silver_Predicted'] = slope * result['Gold_Close'] + intercept
    result['Residual'] = result['Silver_Close'] - result['Silver_Predicted']

    return result

# Trading strategy

def get_treshold(data: pd.DataFrame, entry: float):

    result = data.copy()

    threshold = entry * result["Residual"].std()
    
    # Initialize signals
    result['Signal'] = 0
    
    result.loc[result['Residual'] > threshold, 'Signal'] = -1   # Short silver, long gold
    result.loc[result['Residual'] < -threshold, 'Signal'] = 1   # Long silver, short gold
    
    return result

def run_strategy(data):

    result = data.copy()

    result['Position'] = 0
    position = 0

    for i in range(1, len(result)):
        if result["Signal"].iloc[i] != 0:
            position = result["Signal"].iloc[i]
        elif position != 0 and np.sign(result['Residual'].iloc[i-1]) != np.sign(result['Residual'].iloc[i]):
            """Exit position if the residuals change sign"""
            position = 0  # exit
        result.iloc[i, result.columns.get_loc('Position')] = position

    result['Silver_Return'] = result['Silver_Close'].pct_change()
    result['Gold_Return'] = result['Gold_Close'].pct_change()


    result['Strategy_Return'] = result['Position'].shift(1) * (result['Silver_Return'] - result['Gold_Return'])

    result['Cumulative_Return'] = (1 + result['Strategy_Return']).cumprod()

    return result

def get_batches(data, lenght):

    batches = []
    for i in range(len(data) // lenght - 1):
        batches.append(i * lenght)
    return batches

def obtain_pnl(start_date: str, end_date: str, batch_size: int, entry: float, data: pd.DataFrame):

    result = data.copy()

    batches = get_batches(result, batch_size)

    pnl_list = []

    for i in range(1,len(batches)-1):
        train = result[batches[i-1]:batches[i]]
        test = result[batches[i]:batches[i+1]]

        corr, slope, intercept, r_value, p_value, std_error = get_linear_regression(train)

        test = get_predictions(test, slope=slope, intercept=intercept)

        test = get_treshold(test, entry=entry)

        test = run_strategy(test)

        pnl = test['Cumulative_Return'][-1]

        year_adjusted_pnl = pnl ** (252 / len(test))

        pnl_list.append(year_adjusted_pnl)
    
    return pnl_list
    

if __name__ == "__main__":
    start_date = "2005-01-01"
    end_date = datetime.date.today() # 27/07/2025

    BATCH_SIZE = 365

    tresholds = np.arange(0.05, 2.55, 0.05)
    mean_return = []
    data = get_data(start_date=start_date, end_date=end_date)
    new_data = data.copy()
    for treshold in tresholds:

        pnl_list = obtain_pnl(start_date=start_date, end_date=end_date, batch_size=BATCH_SIZE,
                entry=treshold, data = new_data)
        
        mean_return.append(np.mean(pnl_list))
    
    print(f"The maximum mean return ocurs with a threshold of {tresholds[np.argmax(mean_return)]}\n\
          This gives a return of: {np.max(mean_return)}")


    new_tresholds = np.arange(1.75, 2.60, 0.05)
    batch_sizes = np.arange(30, 900, 30) # From monthly to 2.5 years (aprox)

    for treshold in new_tresholds:
        mean_return = []
        for batch_size in batch_sizes:
            new_data = data.copy()

            pnl_list = obtain_pnl(start_date=start_date, end_date=end_date, batch_size=batch_size,
                    entry=treshold, data = new_data)
        
            mean_return.append(np.mean(pnl_list))
        
        print(f"Given a threshold of {treshold}\n\
              the maximum return is given by the batch with size {batch_sizes[np.argmax(mean_return)]}\n\
                yielding a return of: {np.max(mean_return)}")
        


    





        




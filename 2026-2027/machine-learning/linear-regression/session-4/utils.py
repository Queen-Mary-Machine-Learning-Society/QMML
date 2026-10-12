import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


def get_alpha_and_beta(stock: str, index: str, start: str,
        end: str, interval: str = '1d', plot: bool = False) -> torch.Tensor:
    
    index = yf.download(index, start=start, end=end, interval=interval)
    index = index['Close']
    index['pct_change'] = index.pct_change()
    stock = yf.download(stock, start=start, end=end, interval=interval)
    stock = stock['Close']
    stock['pct_change'] = stock.pct_change()

    linear_regression = pd.DataFrame({
    'r_s': stock['pct_change'],
    'r_i': index['pct_change']
    }).dropna()

    lr_np = linear_regression.to_numpy()
    x = torch.tensor(lr_np[:, 1], dtype=torch.float32).unsqueeze(1)
    ones = torch.ones((x.shape[0],1), dtype=torch.float32)
    x = torch.cat((ones, x), dim=1)
    y = torch.tensor(lr_np[:, 0], dtype=torch.float32)

    beta_v = torch.inverse(x.T@x)@x.T@y

    alpha, beta = beta_v[0].item(), beta_v[1].item()

    if plot:
        plot_x = np.linspace(index['pct_change'].min(), index['pct_change'].max(), 1000)
        plot_y = alpha + beta*plot_x
        plt.scatter(index['pct_change'], stock['pct_change'])
        plt.plot(plot_x, plot_y, color='r')
        plt.grid(True)
        plt.show()
 
    return alpha, beta
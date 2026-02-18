import numpy as np
from typing import Union

def pnl(portfolio: np.ndarray) -> np.ndarray:
    "Compute profit and loss as: final_price - initial_value"
    return portfolio[...,-1] - portfolio[...,0]

def compute_returns(portfolio: np.ndarray) -> np.ndarray:
    "Returns an array [P_1/P_0, P_2/P_1, ...]"
    return portfolio[...,1:] / portfolio[...,:-1] - 1

def compute_log_returns(portfolio: np.ndarray) -> np.ndarray:
    return np.log(portfolio[...,1:] / portfolio[...,:-1])

def compute_cumulative_returns(portfolio: np.ndarray) -> np.ndarray:
    return portfolio[...,1:] / portfolio[...,0] - 1


def mean_return(returns: np.ndarray) -> np.ndarray:
    return np.mean(returns, axis=-1)

def volatiliy(returns: np.ndarray, ddof=1) -> np.ndarray:
    "Set ddof = 1 for sample and ddof = 0 for population"
    return np.std(returns, axis=-1, ddof=ddof)

def compute_excess_returns(returns: np.ndarray, rf: Union[float, np.ndarray]) -> np.ndarray: # rf: float | np.ndarray python >= 3.10

    if np.isscalar(rf) or isinstance(rf, (int, float)):
        """TO DO: Adapt for time-varying risk-free rate"""
        return returns - rf
    else:
        print("Only accepting single rf rate in v0")
        return None

def sharpe_ratio(returns: np.ndarray, rf: Union[float, np.ndarray] = 0.0, 
                 dt = 252, annualize = True) -> np.ndarray:
    excess = compute_excess_returns(returns, rf)
    mean_ex = mean_return(excess)
    vol = volatiliy(excess)

    with np.errstate(divide='ignore', invalid='ignore'):
        sharpe = mean_ex / vol
        sharpe = np.where(vol == 0, np.nan, sharpe)

    if annualize:
        sharpe *= np.sqrt(dt)
    return sharpe

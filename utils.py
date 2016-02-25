import numpy as np
import pandas as pd


def fourier(y, freq, k):
    """
    Computes fourier series for seasonality.

    Parameters
    ----------
    y: TimeSeries
    freq: str like 'ab'
        a = ts freq
        b = seasonal period
        allowed values: DA, DW, MM, WW
    m: float
        period length
    k: int
        number of fourier terms

    Example
    -------
    >>> ndays = 800
    >>> date_index = pd.date_range("1/1/2014", periods=800, freq="D")
    >>> daily = pd.Series(np.arange(800), index=date_index)
    >>> weekly_terms = fourier(daily, "DW", 5)
    >>> annual_terms = fourier(daily, "DA", 5)
    """

    days_per_year = 365.25
    days_per_week = 7.0
    if freq == "DA":
        t = y.index.dayofyear/days_per_year
    elif freq == "DW":
        t = (y.index.dayofweek + 1)/days_per_week
    elif freq == "MM":
        months_per_year = 12.0
        t = (y.index.month + 1)/months_per_year
    elif freq == "WW":
        weeks_per_year = days_per_year/days_per_week
        t = (y.index.week + 1)/weeks_per_year
    else:
        raise NotImplementedError("frequency %s not supported." % freq)

    n = len(y) 
    krange = np.arange(1, k + 1)
    X = np.tile(krange, (n, 1))
    X = 2*np.pi*(X.T*t).T
    res = np.c_[np.sin(X), np.cos(X)]
    cols = ["fourier_sin_%d" % i for i in krange] + \
           ["fourier_cos_%d" % i for i in krange]
    return pd.DataFrame(res, index=y.index, columns=cols)



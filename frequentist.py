from __future__ import absolute_import
import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import multiprocessing as mp
import numba as nb
import time
import sklearn.linear_model as lm
from arimax.utils import fourier


ncores = mp.cpu_count()


class ARIMA(object):
    
    def __init__(self, p, d=0, fit_dates=None):
        """
        Parameters
        ----------
        p: int
            Number of lags to fit.
        d: int
            Number of differences
        fit_dates: (start_date, end_date)
        """
        self.p = p
        self.d = d

        self._exog = None
        self._dy = None
        self._y = None
        self._X = None
        self._freq = None
        self._clf = None

        self.aic_ = None
        self.sigma2_ = None
        self.coef_ = None
        self.resid_ = None
        self.fit_dates = fit_dates

    @property
    def _lags(self):
        lags = xrange(1, self.p + 1)
        return {"AR%d" % lag: lag for lag in lags}
        
    def _set_dmatrices(self, y, exog):
        self._freq = y.index.freq
        if self.d:
            dy = y.diff(self.d)
        else:
            dy = y.copy()

        endog = pd.DataFrame(index=y.index, columns=self._lags.keys())
        for lagname, lag in self._lags.iteritems():
            endog[lagname] = dy.shift(lag)
        
        if exog is not None:
            data = pd.concat([endog, exog], axis=1)
        else:
            data = endog
        
        data["y"] = y
        data["dy"] = dy
        data = data.dropna()
        if self.fit_dates is not None:
            fit_dates = map(pd.to_datetime, self.fit_dates)
            data = data.loc[fit_dates[0]:fit_dates[1]]
        self._X = data.drop(["dy", "y"], axis=1)
        self._dy = data["dy"]
        self._y = data["y"]
        self.fit_dates = data.index
        
    def fit(self, y, exog=None, plot=False):
        """
        Parameters
        ----------
        y: TimeSeries
        exog: TimeSeries DataFrame
        """
        self._set_dmatrices(y, exog)
        clf = lm.LinearRegression()
        clf.fit(self._X, self._dy)
        self._clf = clf
        self.coef_ = pd.Series(clf.coef_, index=self._X.columns)
        dy_hat = clf.predict(self._X)
        self.resid_ = self._dy - dy_hat
        rss = np.sum(self.resid_**2)
        self.aic_ = 2*self._X.shape[1] + self._X.shape[0]*np.log(rss)
        self.sigma2_ = rss/(self._X.shape[0] - self._X.shape[1] - 1)
        if plot:
            plt.plot(self._dy.index, self._dy, label="y")
            plt.plot(self._dy.index, dy_hat, label="y_hat")
            plt.legend()
            plt.show()
    
    def predict(self, start_date, end_date, exog=None, nsims=0, njobs=1):
        """
        Parameters
        ----------
        start_date: str
        end_date: str
        exog: TimeSeries DataFrame
        nsims: int

        Returns
        -------
        TimeSeries DataFrame
        """
        start_date, end_date = pd.to_datetime((start_date, end_date))
        
        offset = pd.tseries.frequencies.to_offset(self._freq)
        y_pred = self._y.loc[:(start_date - offset)].copy()
        if self.d > 0:
            dy_pred = y_pred.diff(self.d)
        else:
            dy_pred = y_pred.copy()
        
        dates = pd.date_range(self._y.index.min(), end_date, freq=self._freq)
        y_pred = y_pred.reindex(dates).values
        dy_pred = dy_pred.reindex(dates).values
        maxlag = max(self._lags.values())

        i_pred_dates = np.arange(dates.get_loc(start_date), dates.get_loc(end_date) + 1)
        if exog is not None:
            i_exog_dates = np.arange(exog.index.get_loc(start_date), exog.index.get_loc(end_date) + 1)
        else:
            i_exog_dates = i_pred_dates
        int_offset = i_pred_dates[0] - dates.get_loc(start_date - offset)
        i_dates = np.c_[i_pred_dates, i_exog_dates]

        if nsims:
            nruns = nsims
            sigma = np.sqrt(self.sigma2_)
        else:
            nruns = 1
            sigma = 0.0

        args = (y_pred.copy(), dy_pred.copy(), i_dates,
                exog, self._clf.intercept_, self._clf.coef_, sigma, 
                int_offset, self.d, maxlag)
        npreds = len(i_pred_dates)*nruns
        min_mp_runs = 10
        njobs = ncores if njobs < 0 else njobs
        use_mp = (njobs > 1) and (nruns > min_mp_runs) and (npreds > 1000)
        if not use_mp:
            preds = [_predict(*args) for _ in xrange(nruns)]
        else:
            pool = mp.Pool(processes=ncores)
            preds = [pool.apply_async(_predict, args=args) for _ in xrange(nruns)]
            preds = [p.get() for p in preds]

        preds = pd.DataFrame(np.array(preds).T, index=dates)
        preds = preds.loc[start_date:end_date]
        return preds


def _predict(y_pred, dy_pred, i_dates,
             exog, intercept_, coef_, sigma, int_offset, d, maxlag):
    """
    Factored out of ARIMA so that we can use multiprocessing.
    """

    for iyt, ixt in i_dates:
        x_endog = y_pred[iyt - int_offset*maxlag:iyt - int_offset + 1]
        if exog is not None:
            x_exog = exog.iloc[ixt].values.reshape(-1, )
            x = np.r_[x_endog, x_exog]
        else:
            x = x_endog

        dy_hat = intercept_ + np.dot(coef_, x)
        u = sigma*np.random.randn()
        dy_pred[iyt] = dy_hat + u

        if d > 0:
            y_pred[iyt] = dy_pred[iyt] + y_pred[iyt - d*int_offset]
        else:
            y_pred[iyt] = dy_pred[iyt]

    return y_pred


def auto_arima(y, p_max, exog=None):
    """
    Choose lag length to minimize AIC.

    Parameters
    ----------
    y: TimeSeries
    p_max: int
        maximum lag to consider
    exog: TimeSeries DataFrame
    """
    offset = pd.tseries.frequencies.to_offset(y.index.freq)
    fit_start = y.index[0] + (p_max + 1)*offset
    fit_end = y.index[-1]
    fit_dates = (fit_start, fit_end)
    lowest_aic = np.inf
    best_model = None
    for p in xrange(1, p_max + 1):
        model = ARIMA(p=p, d=0, fit_dates=fit_dates)
        model.fit(y, exog)
        if model.aic_ < lowest_aic:
            lowest_aic = model.aic_
            best_model = model
    return best_model


def seasonal_auto_arima(y, p_max, fourier_args, exog=None):
    """
    Choose lag length and number of fourier terms to minimize AIC.

    Parameters
    ----------
    y: TimeSeries
    p_max: int
    fourier_args = [(freq, kmax)]
    """
    freqs = [f for (f, _) in fourier_args]
    ranges = [xrange(1, k + 1) for (_, k) in fourier_args]
    lowest_aic = np.inf
    best_kcombo = None
    best_p = None
    for kcombo in it.product(*ranges):
        seasonality = pd.concat([fourier(y, f, k) 
                                 for (f, k) in it.izip(freqs, kcombo)], axis=1)
        if exog is not None:
            exog = pd.concat([exog, seasonality], axis=1)
        else:
            exog = seasonality

        model = auto_arima(y, p_max, exog=exog)
        if model.aic_ < lowest_aic:
            lowest_aic = model.aic_
            best_kcombo = kcombo
            best_p = model.p
    best_fourier = [(freq, k) for (freq, k) in it.izip(freqs, best_kcombo)]
    return best_p, best_fourier


def timed(f):
    def _f(*args):
        s = time.clock()
        res = f(*args)
        t = time.clock()
        elapsed = 1e3*(t - s)
        print "elapsed = %0.0f ms" % elapsed
        return res
    return _f


def benchmark():
    N = 10000

    xs = pd.Series(np.random.randn(N), index=pd.date_range("1/1/2015", periods=N)) 
    arima = ARIMA(p=20)
    arima.fit(xs)
    npreds = 1000000
    start_date, end_date = "1/1/2017", "1/1/2026"
    ndates = len(pd.date_range(start_date, end_date))
    nsims = int(1.0*npreds/ndates)
    print "npreds, nsims = %d, %d" % (ndates, nsims)

    @timed
    def with_mp():
        return arima.predict(start_date, end_date, nsims=nsims, njobs=-1)

    @timed
    def without_mp():
        return arima.predict(start_date, end_date, nsims=nsims)

    print "with mp:"
    with_mp()

    print "without mp"
    without_mp()


def example():

    # Make seasonal AR(2)
    N = 4000
    y = np.zeros(N)
    sigma = 1.0
    rho1 = 0.7
    rho2 = 0.1
    u = sigma*np.random.randn(N)
    t = np.arange(N)
    s = np.sin(2*np.pi*t/365.25)
    y[:2] = s[:2] + u[:2]
    for t in xrange(2, N):
        y[t] = 0.5*s[t] + rho1*y[t - 1] + rho2*y[t - 2] + u[t]
    y = pd.Series(y, index=pd.date_range("1/1/2014", periods=len(y), freq="D"))
    y_tr = y.loc[:"12/31/2020"]
    y_te = y.loc["12/31/2020":]

    # Fit model
    best_params = seasonal_auto_arima(y=y_tr, p_max=10, fourier_args=[("DA", 10)])
    best_p, best_fourier = best_params

    exog_tr = fourier(y_tr, best_fourier[0][0], best_fourier[0][1])
    exog_te = fourier(y_te, best_fourier[0][0], best_fourier[0][1])

    model = ARIMA(p=best_p)
    model.fit(y_tr, exog_tr)

    y_te_hat = model.predict(start_date=y_te.index[0], end_date=y_te.index[-1], 
                             exog=exog_te, nsims=300, njobs=-1)

    # Plot
    plt.figure()
    #plt.plot(y_te.index, y_te, lw=1.0, label="y", alpha=0.5, color="grey")
    plt.plot(y_te_hat.index, y_te_hat.mean(axis=1), label="y_hat", alpha=1.0)
    lower = np.percentile(y_te_hat.values, 5, axis=1)
    upper = np.percentile(y_te_hat.values, 95, axis=1)
    plt.fill_between(y_te_hat.index, lower, upper, alpha=0.25, facecolor="green", interpolate=True)
    plt.xticks(rotation="90")
    plt.tight_layout()

    plt.legend()
    plt.show()


if __name__ == "__main__":
    example()

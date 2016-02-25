import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import functools
import multiprocessing as mp
import re

ncores = mp.cpu_count()


def is_oob(xs, index, periods):
    return (index - periods) < 0 or (index - periods) >= len(xs)


def _lag(xs, index, periods):
    if is_oob(xs, index, periods):
        return np.nan
    else:
        return xs.iat[index - periods]


def _movav(xs, index, periods):
    index += 1
    if is_oob(xs, index, periods):
        return np.nan
    return xs.iloc[(index - periods):index].mean()


def ar(periods):
    """
    Autoregressive function.  See DMatrices documentation for usage.

    Parameters
    ----------
    periods: int
        Number of periods to lag (if positive) or lead (if negative)
    """
    return functools.partial(_lag, periods=periods)


def movav(periods):
    """
    Not used yet.
    """
    return functools.partial(_movav, periods=periods)


class DMatrices(object):
    """
    Manages data for time series regressions.

    Parameters
    ----------
    endog: pd.Series
    xforms: {var_name: fcn}
    exog: pd.DataFrame
    fcst_horizon (default = 1): int
        Number of periods to forecast each time we get a new value of y.

    Example
    -------
    >>> # create random walk
    >>> ys = pd.Series(np.cumsum(np.random.randn(100)), index=pd.date_range("1/1/2015", periods=100))
    >>> # create AR(1) and AR(2) features.
    >>> xforms = dict(ar1=ar(1), ar2=ar(2))
    >>> dm = DMatrices(endog=ys, xforms=xforms)
    """

    def __init__(self, endog, xforms, exog=None, fcst_horizon=1):
        self.xforms = xforms

        self._endog = endog.copy()
        exog = exog.copy() if exog is not None else exog
        self._exog = exog
        self.fcst_horizon = fcst_horizon

        self._fit_slice = None
        self._X = None
        self._initialize_X()
        self._endog_len = len(self._endog)

    @property
    def freq(self):
        return self._endog.index.freq

    def _update_row(self, date, y):
        """
        Updates a row of X, given a new endogenous (date, value) pair.

        Parameters
        ----------
        date: date or str
        y: float
        """

        self._X, self._endog = _mp_update_row(dm=self, X=self._X, y=self._endog, update_date=date, update_y=y)

    def _date2index(self, date):
        return _mp_date2index(self._endog, date)

    def _initialize_X(self):
        self._X = pd.DataFrame(index=self._endog.index, columns=self.xforms.keys())
        next_date = None
        # Fill in autoregressive values until we reach a nan y value.
        for date in self._X.index:
            if np.isnan(self._endog.loc[date]):
                next_date = date
                break
            self._update_row(date=date, y=self._endog.loc[date])

        # After we reach a nan y value, fill in the autoregressive values for the next period.
        if next_date is not None:
            next_date_index = self._date2index(next_date) + 1
            if next_date_index < len(self._endog):
                self._update_row(date=next_date, y=self._endog.loc[next_date])

        self._X = pd.concat([self._X, self._exog], axis=1)

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._endog

    def update(self, date, y):
        """
        Update the data set with a new (date, value) pair for y.

        Parameters
        ----------
        date: date or str
        y: float
        """
        self._update_row(date=date, y=y)

    @property
    def fit_slice(self):
        if self._fit_slice is None:
            fit_data = pd.concat([self.X.copy(), self._endog], axis=1)
            col_sum = fit_data.sum(axis=1, skipna=False).dropna()
            self._fit_slice = slice(col_sum.index.min(), col_sum.index.max())

        return self._fit_slice

    @property
    def fcst_slice(self):
        fcst_start_index = self._endog.index.get_loc(self.fit_slice.stop) + 1
        if fcst_start_index < len(self._endog):
            fcst_start = self._endog.index[fcst_start_index]
            fcst_end = self._endog.index[-1]
            return slice(fcst_start, fcst_end)
        else:
            return None

    @property
    def fit_data(self):
        return self.X.ix[self.fit_slice, :], self.y.ix[self.fit_slice]

    @property
    def fcst_data(self):
        return self.X.ix[self.fcst_slice, :], self.y.ix[self.fcst_slice]

    def get_X_row(self, date):
        return _mp_get_X_row(self.X, date)


def _mp_date2index(xs, date):
    """
    Find the index of a time series xs for a given date.

    Parameters
    ----------
    xs: TimeSeries
    date: date

    Returns
    -------
    int
    """
    return xs.index.get_loc(date)


def _mp_update_row(dm, X, y, update_date, update_y, nsteps=0):
    """
    This function belongs in DMatrices, but I factored it out so I could use multiprocessing.

    For a given (update_date, update_y) pair, update X and y until nsteps=dm.fcst_horizon

    Parameters
    ----------
    dm: DMatrices instance
    X: TimeSeries DataFrame
    y: TimeSeries series
    update_date: date
    update_y: float
    nsteps: int
    """
    cur_date = update_date
    cur_date_index = _mp_date2index(y, cur_date)
    next_date_index = cur_date_index + 1

    y.set_value(update_date, update_y)
    for var_name, fcn in dm.xforms.iteritems():
        val = fcn(y, _mp_date2index(y, update_date))
        X.set_value(update_date, var_name, val)

    if (nsteps >= dm.fcst_horizon) or (next_date_index >= len(y)):
        return X, y

    next_date = y.index[next_date_index]
    return _mp_update_row(dm=dm, X=X, y=y, update_date=next_date, update_y=y.at[next_date], nsteps=nsteps + 1)


def _mp_get_X_row(X, date):
    """
    Parameters
    ----------
    X: TimeSeries DataFrame
    date: date

    Returns
    -------
    Series
    """
    return X.loc[date, :]


def _mp_predict(dm, w, sig_y, fcst_dates):
    """
    Recursively predict y.

    Parameters
    ----------
    dm: DMatrices
    w: np.array of weights
    sig_y: float
    fcst_dates: pd.date_range
    """
    X, y = dm.X.copy(), dm.y.copy()
    for t, date in enumerate(fcst_dates):
        y_hat_t = np.dot(_mp_get_X_row(X, date), w) + sig_y*np.random.randn()
        if np.isnan(y_hat_t):
            raise ValueError("non numeric prediction")
        X, y = _mp_update_row(dm=dm, X=X, y=y, update_date=date, update_y=y_hat_t)
    return y.loc[fcst_dates]


class DynamicRegression(object):
    """
    Bayesian ridge time series regression and prediction.

    Model:
        y = Xw + u
        w ~ N(0, alpha**(-2)*np.eye(K))
        u ~ N(0, sigma**2*np.eye(N))
        X includes lagged values for y

    See fit() documentation for more detail.
    """

    def __init__(self, nsims=5000, burn_rate=0.5):
        self.nsims = nsims
        self.burn_rate = burn_rate
        self.dmatrices = None
        self.mcmc = None
        self.fit_start = None
        self.fit_end = None

    @staticmethod
    def _ols(X, y):
        XX = X.T.dot(X)
        invXX = np.linalg.pinv(XX)
        w = invXX.dot(X.T).dot(y)
        y_hat = X.dot(w)
        u_hat = y - y_hat
        sig2 = np.sum(u_hat**2)/(X.shape[0] - X.shape[1])
        V_w = sig2*invXX
        sig = np.sqrt(sig2)
        return w, sig, np.sqrt(np.diag(V_w))

    @staticmethod
    def _make_model(X, y, w_start, sig_y_start, alpha_start):
        N, K = X.shape
        alpha = pm.Uniform(name="alpha", lower=0.0, upper=1e4, value=alpha_start)

        @pm.deterministic
        def tau_alpha(_alpha=alpha):
            return np.diag(_alpha**(-2.0))

        w = pm.MvNormal(name="w", mu=np.zeros(K), tau=tau_alpha, value=w_start)

        @pm.deterministic
        def y_hat(_w=w):
            return X.dot(_w)

        sig_y = pm.Uniform(name="sig_y", lower=0.0, upper=1e4, value=sig_y_start)
        tau_y = sig_y**(-2.0)
        y_like = pm.Normal(name="y_like", mu=y_hat, tau=tau_y, value=y, observed=True)

        return locals()

    def set_dmatrices(self, y, xforms, exog, fit_start=None, fit_end=None):
        self.dmatrices = DMatrices(endog=y, xforms=xforms, exog=exog)
        self.fit_start = fit_start or self.dmatrices.fit_slice.start
        self.fit_end = fit_end or self.dmatrices.fit_slice.stop

    def fit(self, y, xforms, exog=None, fit_start=None, fit_end=None):
        """
        Parameters
        ----------
        y: TimeSeries Series
            y's values should be like [float, ..., float] + [np.nan, ..., np.nan]
            the forecast will automatically start at the first nan and end at the last date.
        xforms: {var_name: fcn}
        exog: TimeSeries DataFrame
            For the forecast, there mustn't be any exogenous nan's.
        nsims: number of mcmc simulations
        burn_rate: percentage of simulations to burn
        fit_start (optional): date
        fit_end (optional): date

        Example
        -------
        >>> # Create a random walk.  Fit an AR(2) model with a trend.  Plot the results.
        >>> # create random walk
        >>> ys = pd.Series(np.cumsum(np.random.randn(100)), index=pd.date_range("1/1/2015", periods=100))
        >>> # We'll forecast 3/1/2015-onward.
        >>> ys["3/1/2015":] = np.nan
        >>> # add features for an AR(2) model
        >>> xforms = dict(ar1=ar(1), ar2=ar(2))
        >>> # add a trend term
        >>> exog = pd.DataFrame(np.arange(len(ys), index=ys.index, columns=["trend"])
        >>> dr = DynamicRegression()
        >>> dr.fit(y=ys, xforms=xforms, exog=exog)
        >>> dr.plot_fit()
        >>> dr.plot_fcst()
        """
        self.set_dmatrices(y=y, xforms=xforms, exog=exog, fit_start=fit_start, fit_end=fit_end)
        X, y = self.dmatrices.fit_data
        w_start, sig_start, alpha_start = self._ols(X, y)
        mcmc = pm.MCMC(self._make_model(X=X.values, y=y.values, w_start=w_start,
                                        sig_y_start=sig_start, alpha_start=alpha_start), verbose=0)
        mcmc.sample(self.nsims, burn=int(self.burn_rate*self.nsims))
        self.mcmc = mcmc

    def plot_fit(self):
        X, y = self.dmatrices.fit_data
        w_hat = np.median(self.mcmc.w.trace(), axis=0)
        y_hat = X.dot(w_hat)
        plt.plot(X.index, y)
        plt.plot(X.index, y_hat)
        plt.show()

    def _load_predict_data(self, nsims):
        w = self.mcmc.w.trace()
        sig_y = self.mcmc.sig_y.trace()
        sample_index = np.random.choice(w.shape[0], nsims)
        fcst_dates = pd.date_range(start=self.dmatrices.fcst_slice.start, end=self.dmatrices.fcst_slice.stop,
                                   freq=self.dmatrices.freq)
        return w, sig_y, sample_index, fcst_dates

    def _profilable_predict(self, nsims):
        """
        Single-process predict for profiling purposes.
        """
        w, sig_y, sample_index, fcst_dates = self._load_predict_data(nsims)
        res = [_mp_predict(self.dmatrices, w[j], sig_y[j], fcst_dates) for j in sample_index]
        output = {i: r for i, r in enumerate(res)}
        return pd.DataFrame(output)

    def predict(self, nsims=100, njobs=-1):
        """
        Parameters
        ----------
        nsims: int

        Returns
        -------
        DataFrame(index=[forecast dates], columns=[sim_1, ..., sim_nsims])
        """
        njobs = ncores if njobs < 0 else njobs
        w, sig_y, sample_index, fcst_dates = self._load_predict_data(nsims)
        pool = mp.Pool(processes=njobs)
        res = [pool.apply_async(_mp_predict, args=(self.dmatrices, w[j], sig_y[j], fcst_dates))
               for j in sample_index]
        output = {i: p.get() for i, p in enumerate(res)}
        return pd.DataFrame(output)

    def plot_fcst(self, nsims=100):
        percentiles = [5, 50, 95]
        _, actual = self.dmatrices.fit_data
        preds = self.predict(nsims=nsims)
        lower, middle, upper = [np.percentile(preds, p, axis=1) for p in percentiles]
        plt.plot(actual.index, actual, c="indianred", lw=2.0)
        plt.plot(preds.index, middle, c="royalblue", lw=2.0)
        plt.fill_between(preds.index, lower, upper, color="darkseagreen", alpha=0.6)
        plt.show()

    def is_stationary(self, w):
        def extract_lag(s):
            lag = re.findall("^ar(\d+)$", s)
            return int(lag[0]) if lag else None

        arlags = map(extract_lag, self.colnames)
        arcoefs = {lag: coef for lag, coef in zip(arlags, w) if lag}
        poly = [1] + [-arcoefs[lag] if lag in arcoefs else 0 for lag in range(1, max(arlags + 1))]
        roots = np.roots(poly)
        return max(abs(roots)) < 1.0

    @property
    def colnames(self):
        X, _ = self.dmatrices.fit_data
        return X.columns

    @property
    def coef_(self):
        return pd.Series(np.median(self.mcmc.w.trace(), axis=0), index=self.colnames)

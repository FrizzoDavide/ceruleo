import logging
from typing import Optional

import numpy as np
import pandas as pd
from rul_pm.transformation.transformerstep import TransformerStep, TransformerStepMixin
from sklearn.base import BaseEstimator, TransformerMixin
from tdigest import TDigest

logger = logging.getLogger(__name__)


class PerColumnImputer(TransformerStepMixin, BaseEstimator, TransformerMixin):
    """Impute the values of each column following a simple rule

    The imputing is made following this rule:
        -np.inf -> min
        np.inf -> max
        nan -> median

        Parameters
        ----------
        name : Optional[str], optional
            Step name, by default None
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.data_min = None
        self.data_max = None
        self.data_median = None

    def partial_fit(self, X, y=None):
        X = X.replace([np.inf, -np.inf], np.nan)
        col_to_max = X.max()
        col_to_min = X.max()
        col_to_median = X.median()
        if self.data_min is None:
            self.data_min = col_to_min
            self.data_max = col_to_max
            self.data_median = col_to_median
        else:
            self.data_min = pd.concat([self.data_min, col_to_min],
                                      axis=1).min(axis=1)
            self.data_max = pd.concat([self.data_max, col_to_max],
                                      axis=1).max(axis=1)
            self.data_median = pd.concat([self.data_max, col_to_median],
                                         axis=1).median(axis=1)
        self._remove_na()

    def _remove_na(self):
        self.data_max.fillna(0, inplace=True)
        self.data_min.fillna(0, inplace=True)
        self.data_median.fillna(0, inplace=True)

    def fit(self, X, y=None):
        X = X.replace([np.inf, -np.inf], np.nan)
        col_to_max = X.max()
        col_to_min = X.max()
        col_to_median = X.median()

        self.data_min = col_to_min
        self.data_max = col_to_max
        self.data_median = col_to_median

        self._remove_na()

    def transform(self, X, y=None):
        X_new = X.copy()
        for c in X_new.columns:
            X_new[c] = X_new[c].replace([np.inf], self.data_max[c])
            X_new[c] = X_new[c].replace([-np.inf], self.data_min[c])
            X_new[c] = X_new[c].replace([np.nan], self.data_median[c])
        return X_new


class PandasRemoveInf(TransformerStep):
    """Replace NaN for inf    
    """
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transform the input life replacing Nan for inf

        Parameters
        ----------
        X : pd.DataFrame
            Input Dataframe to be transformed

        Returns
        -------
        pd.DataFrame
            A dataframe with she same index as the input without NaN values
        """
        return X.replace([np.inf, -np.inf], np.nan)


class PandasMedianImputer(TransformerStep):
    """Impute missing values with the median value of the training set

    Parameters
    ----------
    name : Optional[str]
        The name of the step
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.tdigest_dict = None

    def fit(self, X, y=None):
        """Compute the median value 

        Parameters
        ----------
        X : pd.DataFrame
            The input life
        """
        self.median = X.median(axis=0).to_dict()
        return self

    def partial_fit(self, X):
        """Compute the median value incrementally

        Parameters
        ----------
        X : pd.DataFrame
            The input life
        """
        if self.tdigest_dict is None:
            self.tdigest_dict = {c: TDigest() for c in X.columns}
        for c in X.columns:
            self.tdigest_dict[c].batch_update(X[c].values)

        self.median = {
            c: self.tdigest_dict[c].percentile(50)
            for c in self.tdigest_dict.keys()
        }

    def transform(self, X, y=None):
        """Return a new dataframe with the missing values replaced by the fitted median

        Parameters
        ----------
        X : pd.DataFrame
            Life
        

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the same index as the input with the Na values replaced
        """
        return X.fillna(value=self.median)


class PandasMeanImputer(TransformerStep):
    """Impute missing values with the mean value of the training set

    Parameters
    ----------
    name : Optional[str]
        The name of the step
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.sum = None

    def partial_fit(self, X, y=None):
        """Compute the mean value incrementally

        Parameters
        ----------
        X : pd.DataFrame
            The input life
        """
        if self.sum is None:
            self.sum = X.sum(axis=0)
            self.counts = X.shape[0]
        else:
            self.sum += X.sum(axis=0)
            self.counts += X.shape[0]
        self.mean = (self.sum / self.counts).to_dict()
        return self

    def fit(self, X, y=None):
        """Compute the mean value 

        Parameters
        ----------
        X : pd.DataFrame
            The input life
        """
        self.mean = X.mean(axis=0).to_dict()
        return self

    def transform(self, X:pd.DataFrame, y=None) -> pd.DataFrame:
        """Return a new dataframe with the missing values replaced by the fitted mean

        Parameters
        ----------
        X : pd.DataFrame
            Life
        

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the same index as the input with the Na values replaced
        """
        return X.fillna(value=self.mean)


class RollingImputer(TransformerStep):
    """Impute missing values using a function over a rolling window

    Parameters
    ----------
    
    window_size : int
        Window size of the rolling window
    
    func: Callable
        The function to call in each window
    """
    def __init__(self, window_size: int, func):
        self.window_size = window_size
        self.function = func
        self.mean_value_list = []
        self.sum = None

    def partial_fit(self, X:pd.DataFrame, y=None):
        """Compute incrementally the mean value to use as default value to impute

        Parameters
        ----------
        X : pd.DataFrame
            The input lfie
        """
        if self.sum is None:
            self.sum = X.sum(axis=0)
            self.counts = X.shape[0]
        else:
            self.sum += X.sum(axis=0)
            self.counts += X.shape[0]
        self.default_value = (self.sum / self.counts).to_dict()
        return self

    def fit(self, X:pd.DataFrame, y=None):
        """Compute a default value in case there are not valid values in the rolling window

        Parameters
        ----------
        X : pd.DataFrame
            The input life
        """
        self.default_value = np.mean(X, axis=0)
        self.default_value[~np.isfinite(self.default_value)] = 0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input life

        Parameters
        ----------
        X : pd.DataFrame
            The input life to be transformed

        Returns
        -------
        pd.DataFrame
            A new life with the same index as the input with the missing values
            replaced by the output of the function supplied
        """
        X = X.copy()
        row, features = np.where(~np.isfinite(X))
        min_limit = np.maximum(row - self.window_size, 0)
        max_limit = np.minimum(row + self.window_size, X.shape[0])
        for r, min_r, max_r, f in zip(row, min_limit, max_limit, features):
            X[r, f] = self.function(X[min_r:max_r, f])
            if ~np.isfinite(X[r, f]):
                X[r, f] = self.default_value[f]
        return X

    def partial_fit(self, X, y=None):
        return self


class RollingMedianImputer(RollingImputer):
    """Impute missing values with the median value on a rolling window

    Parameters
    ----------
    
    window_size : int
        Window size of the rolling window
    """
    def __init__(self, window_size: int):
        super().__init__(window_size, np.median)


class RollingMeanImputer(RollingImputer):
    """Impute missing values with the mean value on a rolling window

    Parameters
    ----------

    window_size : int
        Window size of the rolling window
    """
    def __init__(self, window_size: int):
        super().__init__(window_size, np.mean)


class ForwardFillImputer(TransformerStep):
    """Impute forward filling the values
    """
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        return X.ffill()

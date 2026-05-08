from typing import Optional, Union, Tuple, List

import ipdb
import numpy as np
from ceruleo.dataset.ts_dataset import AbstractPDMDataset
from ceruleo.dataset.utils import iterate_over_target
from ceruleo.results.results import FittedLife

from ceruleo.dataset.transformed import TransformedDataset


class BaselineModel:
    """
    Predict the RUL using the mean or the median value of the duration of the dataset.
    If the quantile mode is used, passing a quantile level tau will use the tau-th
    quantile of the dataset duration to predict the RUL

    Parameters:
        mode: Method for computing the duration of the dataset. Possible values are: 'mean', 'median' and 'quantile'
        RUL_threshold: RUL RUL_threshold, by default None
        tau: quantile level to use for the quantile model
    """

    def __init__(self, mode: str = "mean", RUL_threshold: Optional[float] = None, tau: float = 0.5):

        self.mode = mode
        self.RUL_threshold = RUL_threshold
        self.tau = tau


    def fit(self, ds: Union[TransformedDataset, AbstractPDMDataset]):
        """Compute the mean or median RUL using the given dataset

        Parameters:
            ds:  Dataset from which obtain the true RUL
        """
        true = []
        for y in iterate_over_target(ds):
            y = y
            degrading_start, time = FittedLife.compute_time_feature(
                y, self.RUL_threshold
            )

            true.append(y.iloc[0] + time[degrading_start])

        if self.mode == "mean":
            self.fitted_RUL = np.mean(true)
        elif self.mode == "median":
            self.fitted_RUL = np.median(true)
        elif self.mode == "quantile":
            self.fitted_RUL = np.quantile(true, self.tau)
        else:
            raise ValueError(f"mode {self.mode} not supported")

    def predict(self, ds: TransformedDataset) -> np.ndarray:
        """
        Predict the whole life using the fitted values

        Parameters:
            ds: Dataset iterator from which obtain the true RUL

        Returns:
            Predicted RUL
        """
        output = []
        for y in iterate_over_target(ds):
            _, time = FittedLife.compute_time_feature(y, self.RUL_threshold)
            y_pred = np.clip(self.fitted_RUL - time, 0, self.fitted_RUL)
            output.append(y_pred)
        return np.concatenate(output)

    def predict_lifes(self, ds: TransformedDataset) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Modification of the predict method to return separated
        predictions and true values over the different lifes

        Parameters:
            ds: Dataset iterator from which obtain the true RUL

        Returns:
            y_pred, y_true (Tuple[List[np.ndarray], List[np.ndarray]): list of predictions and true values over the lifes
        """

        y_pred, y_true = [], []
        for y in iterate_over_target(ds):
            _, time = FittedLife.compute_time_feature(y, self.RUL_threshold)
            pred = np.clip(self.fitted_RUL - time, 0, self.fitted_RUL)
            y_pred.append(pred)
            y_true.append(y)

        return y_pred, y_true

class FixedValueBaselineModel:
    """
    A model that predicts always  the same duration for each run-to-failure cycle

    Parameters:
        value: Fixed RUL
    """

    def __init__(self, *, value: float):
        self.value = value

    def fit(self, *args):
        return self

    def predict(
        self, ds: TransformedDataset, RUL_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict the whole life using the fixed values

        Parameters:
            ds: Dataset iterator from which obtain the true RUL

        Returns:
            Predicted RUL
        """
        output = []
        for y in iterate_over_target(ds):
            _, time = FittedLife.compute_time_feature(y, RUL_threshold)
            y_pred = np.clip(self.value - time, 0, self.value)
            output.append(y_pred)
        return np.concatenate(output)

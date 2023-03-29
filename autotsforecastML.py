# -*- coding: utf-8 -*-
"""autotsforecastML.py

Structure:

class a:
    methode1():
        submethode():
            # some code
            # some code 
        # some code
        
    methode2():
        submethode():
            # some code
            # some code
        # some code
        .
        .
        .
    methode3():
        submethode():
            # some code
            # some code
        # some code
"""
# Load Modules
# system
import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.CRITICAL)

import sys, numbers, math
from time import perf_counter
import datetime as dt

# for data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from functools import reduce
from typing import List, Tuple, Union
from tqdm.auto import tqdm

# forecasting helper
import pmdarima as pmd
import statsmodels.api as sm 
from scipy.stats import normaltest

# darts
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import (
    ARIMA,
    AutoARIMA,
    ExponentialSmoothing,
    NaiveSeasonal,
    NaiveDrift,
    Prophet,
    RegressionEnsembleModel,
    Theta
)
from darts.dataprocessing.transformers.boxcox import BoxCox
from darts.metrics import mape, mase, mae, mse, ope, r2_score, rmse, rmsle
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
from darts.utils.utils import ModelMode
from darts.datasets import ( 
    AirPassengersDataset,
    ElectricityDataset
)

class AutoUnivariateiTS:

    def __init__(
        self,
        trace: bool = False
    ) -> None:

        self.TRACE = trace

    def pandasdf_to_timeriesdata(
        self,
        data: pd.DataFrame,
        target_column: List[str] = [],
        index_col: str = None
    ):
        if not target_column:
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            tseries = TimeSeries.from_dataframe(data, value_cols=data.columns)
        
        elif index_col:
            data = data.copy().set_index(index_col)
            tseries = TimeSeries.from_dataframe(data, value_cols=data.columns)

        else:
            raise ValueError("""
            Invalid index column. Set valid index column using 'index_col' parameter !!!
            """)
        else:
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            tseries = TimeSeries.from_dataframe(data, value_cols=target_column)
        
        elif index_col:
            data = data.copy().set_index(index_col)
            tseries = TimeSeries.from_dataframe(data, value_cols=target_column)

        else:
            raise ValueError("""
            Invalid index column. Set valid index column using 'index_col' parameter !!!
            """)
        self._data = tseries
        return tseries

    def timeriesdata_to_pandasdf(
        self,
        data: darts.timeseries.TimeSeries,
        reset_index: bool = False
    ):
        
        if not reset_index:
        pdf = data.pd_dataframe()
        else:
        pdf = data.pd_dataframe().reset_index()
        
        return pdf

    def timeriesdata_to_pdseries(
        self,
        data: darts.timeseries.TimeSeries,
    ):
        pdseries = data.pd_series()
        return pdseries

    def seasonality_check(
        self,
        data,
        seasonality_start: int = 2,
        seasonality_end: int = 50,
        alpha = 0.05,
        print_summary: bool = True
    ):
        self.ALPHA = alpha
        for m in range(seasonality_start, seasonality_end):
        is_seasonal, MSEAS = check_seasonality(data, m=m, alpha=alpha)
        if is_seasonal:
            break
        self.is_seasonal = is_seasonal
        self.MSEAS = MSEAS
        if print_summary:
        print(f"Is provided data seasonal? :{self.is_seasonal}")
        if self.is_seasonal:
            print(f"There is seasonality of order {self.MSEAS}")  
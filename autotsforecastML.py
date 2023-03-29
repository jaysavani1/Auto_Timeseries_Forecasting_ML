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

    def train_test_split_data(
        self,
        data: darts.timeseries.TimeSeries,
        spliting_at: Union[pd.Timestamp, int, float],
        split_before: bool = True,
        plot: bool = True,
        plot_size = (12, 5),
        legend_loc = 'upper right'
    ):
        # split position: if string, then interpret as Timestamp
        # if int, then interpretation as index
        # if loat, then interpretation as %split

        if isinstance(spliting_at, numbers.Number):
        split_at = spliting_at
        else:
            split_at = pd.Timestamp(spliting_at)
        train, val = data.split_before(split_at)
        if plot:
        plt.figure(101, figsize = plot_size)
        train.plot(label ='training')
        val.plot(label ='validation')
        plt.legend(loc = legend_loc)
        
        return train, val

    def fit_predict(
        self,
        train_data: darts.timeseries.TimeSeries,
        val_data: darts.timeseries.TimeSeries,
        select_model: List[str] = [],
        select_all_models: bool = True,
        seasonality_check: bool = True
    ):
        if (not select_model) and (not select_all_models):
        raise ValueError("""
            'select_model' should not be empty list. Select atleast one model or use 'select_all_models' = True !!!
        """)
        
        # elif select_model and select_all_models:
        #   raise ValueError("""
        #     Can not use both 'select_model' and 'select_all_models' = True at the same time. Parameter 'select_all_models' should be set to False !!!
        #   """)
        # else:
        self.train = train_data
        self.val = val_data
        self.seasonality_check(self._data)
        self.selected_models = self.selecting_models(select_model) if select_model else self.selecting_models()
    
        # Evaluate model performance
        def _run_model(
            model_name: str, 
            model
        ):
        
            pbar.set_description("Processing %s" % model_name)
            t_start =  perf_counter()
            print(f"\n======================={model_name.upper()} - MDOEL SUMMARY=======================")

            print(f"\nModel parameters: {str(model)}")

            # fit the model and compute predictions
            res = model.fit(self.train)
            forecast = model.predict(len(self.val))

            if seasonality_check:
                # for naive forecast, concatenate seasonal fc with drift fc
                if model_name == 'naive drift':
                    if self.is_seasonal:
                        fc_drift = forecast
                        modelS = NaiveSeasonal(K=self.MSEAS)
                        modelS.fit(self.train)
                        fc_seas = modelS.predict(len(self.val))
                        forecast = fc_drift + fc_seas - self.train.last_value()
                res_time = perf_counter() - t_start
            
            print(f"Calculating Error Metrics:..")

            # compute accuracy metrics and processing time
            res_mape = mape(self.val, forecast)
            res_mae = mae(self.val, forecast)
            res_r2 = r2_score(self.val, forecast)
            res_rmse = rmse(self.val, forecast)
            res_rmsle = rmsle(self.val, forecast)
            res_time = perf_counter() - t_start
            res_accuracy = {"MAPE":res_mape, "MAE":res_mae, "R squared":-res_r2, "RMSE":res_rmse, "RMSLE":res_rmsle, "time":res_time}
            results = [forecast, {model_name : res_accuracy}]
            
            print(f"Trial Finished... Total time taken:{res_time} sec")

            return results
    
        pbar = tqdm(self.selected_models.items())
        self._model_predictions = [_run_model(model_name = m_name, model = model) for m_name, model in pbar]
        
        # Prepare Performance Metrics
        res = pd.DataFrame(columns=['MAE', 'MAPE', 'R squared', 'RMSE', 'RMSLE', 'time'])
        for i in range(len(self._model_predictions)):
        res = pd.concat([res, pd.DataFrame(self._model_predictions[i][1]).T])
        pd.set_option("display.precision",3)
        res.style.highlight_min(color="blue", axis=0).highlight_max(color="red", axis=0)
        
        return res
        
    def selecting_models(
        self, 
        model_list: List[str] = None, 
        all: bool = True
    ):

        _DEFAULT_MODELS = {
            'auto arima' : _get_auto_arima(),
            'exponential smoothing' : ExponentialSmoothing(seasonal_periods= self.MSEAS) if self.is_seasonal else ExponentialSmoothing(),
            'theta' : _get_theta(),
            'naive drift' : NaiveDrift(),
            'prophet (additive seasonality)' : Prophet(seasonality_mode='additive'),
            'prophet (multiplicative seasonality)': Prophet(seasonality_mode='multiplicative')
        }

        if all:
        return _DEFAULT_MODELS
        else:
        filter_model_selection = {k:v for k,v in _DEFAULT_MODELS.items() if k in model_list}
        return filter_model_selection
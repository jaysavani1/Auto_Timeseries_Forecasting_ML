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
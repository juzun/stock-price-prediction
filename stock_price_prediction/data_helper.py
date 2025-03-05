import datetime as dt
import warnings
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
from ruptures import Binseg

from stock_price_prediction.types import TickerSymbol


class DataHelper:
    """
    Utility class for fetching, preprocessing, and analyzing stock market data.
    """

    @staticmethod
    def fetch_stock_history(
        tickers: List[TickerSymbol],
        start_date: dt.date = dt.date(1950, 1, 1),
        end_date: dt.date = dt.date(2025, 1, 1),
    ) -> pd.DataFrame:
        """
        Fetches historical stock price data from Yahoo Finance.

        Args:
            tickers (List[TickerSymbol]): A list of stock ticker symbols.
            start_date (dt.date, optional): The start date for fetching data. Defaults to 1950-01-01.
            end_date (dt.date, optional): The end date for fetching data. Defaults to 2025-01-01.

        Returns:
            pd.DataFrame: A DataFrame containing stock price data.
        """
        stock_history = yf.download(
            tickers=[ticker.name for ticker in tickers],
            start=start_date,
            end=end_date,
            auto_adjust=False,
            actions=True,
            multi_level_index=False,
            progress=False,
        )
        if stock_history.empty:
            warnings.warn("Stock history returned from Yahoo Finance is empty.")
        return stock_history

    @staticmethod
    def preprocess_stock_history(stock_history: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and enhances stock price data by computing technical indicators.

        - Computes returns, moving averages, volatility, and RSI.
        - Adds lagged returns for short-term trend tracking.
        - Interpolates missing data points and one-hot encodes the day of the week.

        Args:
            stock_history (pd.DataFrame): The raw stock data.

        Returns:
            pd.DataFrame: The processed stock data with additional features.
        """
        stock_history = stock_history.dropna()

        # Moving Averages
        stock_history["MA_10"] = stock_history["Adj Close"].rolling(window=10).mean()
        stock_history["MA_50"] = stock_history["Adj Close"].rolling(window=50).mean()

        # Drop missing values due to rolling window calculations
        stock_history = stock_history.dropna()

        # Set data frequency to business days and interpolate newly added nan values.
        stock_history = stock_history.asfreq("B").interpolate()

        features_to_keep = ["Adj Close", "Close", "High", "Low", "MA_10", "MA_50"]
        stock_history = stock_history[features_to_keep]

        new_start_date = DataHelper.detect_breakout_point(
            prices=stock_history["Adj Close"]
        ).replace(month=1, day=1)
        stock_history_cut = stock_history[stock_history.index >= new_start_date]

        return stock_history_cut

    @staticmethod
    def detect_breakout_point(prices: pd.Series) -> pd.Timestamp:
        """
        Detects the most significant structural change (breakout point) in a time series.

        Args:
            prices (pd.Series): A time series of stock prices.

        Returns:
            pd.Timestamp: The timestamp of the detected breakout point.
        """
        data_reshaped = np.array(prices).reshape(-1, 1)
        binseg_model = Binseg(model="l2").fit(data_reshaped)
        change_points = binseg_model.predict(n_bkps=1)

        return prices.index[min(change_points)]

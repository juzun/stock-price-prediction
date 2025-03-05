import datetime as dt
import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from keras.api.callbacks import EarlyStopping
from keras.api.layers import LSTM, Dense, Dropout, Input
from keras.api.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler


class LSTMModel:
    """Long Short-Term Memory (LSTM) model for time-series forecasting.

    This class trains an LSTM model using past stock prices and other features to predict future prices.

    Attributes:
        data (pd.DataFrame): The dataset containing historical stock prices and features.
        features (List[str]): The list of feature column names used for training.
        target_feature (str): The target variable for prediction.
        feature_index_map (dict): A mapping from feature names to their index in the dataset.
        seq_length (int): The length of input sequences used for training.
        trained (bool): Indicates whether the model has been trained.
        model (Optional[Sequential]): The LSTM model (None until trained).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        target_feature: str,
        seq_length: int = 30,
    ) -> None:
        """Initializes the LSTM model.

        Args:
            data (pd.DataFrame): The dataset containing stock prices and features.
            features (List[str]): List of feature column names.
            target_feature (str): The target feature for prediction.
            seq_length (int, optional): The number of past days used for prediction. Defaults to 30.
        """
        self.data = data
        self.features = features
        self.target_feature = target_feature
        self.feature_index_map = {feat: idx for idx, feat in enumerate(features)}
        self.seq_length = seq_length
        self.trained = False
        self.model: Optional[Sequential] = None

    def train_model(self) -> None:
        """Trains the LSTM model.

        This method:
        - Splits the dataset into training and test sets.
        - Scales the features.
        - Creates and compiles the LSTM model.
        - Trains the model using early stopping.
        """
        self.split_scale_sequence()

        self.model = self.create_model()
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=20,
            batch_size=32,
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stopping],
            verbose=1,
            sample_weight=np.linspace(0.5, 2, self.X_train.shape[0]),
        )

        self.trained = True

    def split_scale_sequence(self) -> None:
        """Splits the dataset into training and test sets, then scales the features."""
        train_size = int(len(self.data) * 0.8)
        train_data = self.data.iloc[:train_size]
        test_data = self.data.iloc[train_size:]

        # Fit a scaler on training data
        self.scaler = MinMaxScaler()
        train_data_scaled = pd.DataFrame(
            self.scaler.fit_transform(train_data[self.features]),
            columns=self.features,
            index=train_data.index,
        )
        self.X_train, self.y_train = self.create_sequences(
            data=train_data_scaled,
        )

        # Transform test data
        self.test_data_scaled = pd.DataFrame(
            self.scaler.transform(test_data[self.features]),
            columns=self.features,
            index=test_data.index,
        )
        self.X_test, self.y_test = self.create_sequences(
            data=self.test_data_scaled,
        )

    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Creates sequences for LSTM training.

        Args:
            data (pd.DataFrame): The scaled dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Sequences (X) and corresponding target values (y).
        """
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data.iloc[i : i + self.seq_length].values)
            y.append(
                data.iloc[
                    i + self.seq_length, self.feature_index_map[self.target_feature]
                ]
            )
        return np.array(X), np.array(y)

    def create_model(self) -> Sequential:
        """Creates an LSTM model architecture.

        Returns:
            Sequential: A compiled LSTM model.
        """
        return Sequential(
            [
                Input(shape=(self.seq_length, len(self.features))),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation="relu"),
                Dense(1),
            ]
        )

    def make_predictions(
        self,
        days_ahead: int,
    ) -> np.ndarray:
        """Generates predictions for a specified number of future days.

        Args:
            days_ahead (int): The number of future days to predict.

        Returns:
            np.ndarray: The predicted stock prices (smoothed).
        """
        if self.model is None:
            warnings.warn("The LSTM model wasn't yet created.")
            return np.ndarray(0)

        # Start with the most recent known sequence
        if isinstance(self.test_data_scaled, pd.DataFrame):
            input_seq = self.test_data_scaled.to_numpy()[-self.seq_length :].copy()
        else:
            input_seq = self.test_data_scaled[-self.seq_length :].copy()

        predictions = []

        for _ in range(days_ahead):
            predicted_value = self.model.predict(
                input_seq.reshape(1, self.seq_length, -1), verbose=0
            )[0, 0]
            predictions.append(predicted_value)

            # Update sequence by appending prediction and shifting window
            new_entry = np.roll(input_seq, shift=-1, axis=0)
            new_entry[-1] = input_seq[-1]
            new_entry[-1, self.feature_index_map[self.target_feature]] = predicted_value
            input_seq = new_entry

        # Rescale predictions
        predictions_all_fetures = np.zeros((len(predictions), len(self.features)))
        predictions_all_fetures[:, self.feature_index_map[self.target_feature]] = (
            predictions
        )
        predictions_descaled = self.scaler.inverse_transform(predictions_all_fetures)[
            :, self.feature_index_map[self.target_feature]
        ]

        predictions_with_last_day = np.insert(
            predictions_descaled, 0, self.data[self.target_feature].iloc[-1]
        )

        # Apply smoothing
        smoothing_factor = 0.1
        predictions_smoothed = np.array([predictions_with_last_day[0]])
        for p in predictions_with_last_day[1:]:
            predictions_smoothed = np.append(
                predictions_smoothed,
                (
                    smoothing_factor * p
                    + (1 - smoothing_factor) * predictions_smoothed[-1]
                ),
            )

        return predictions_smoothed

    def get_model_statistics(self) -> dict:
        """Evaluates the model on test data and returns performance metrics.

        Returns:
            dict: A dictionary containing MSE, MAE, and benchmark comparisons.
        """
        if self.model is None:
            warnings.warn("The LSTM model wasn't yet created.")
            return {}

        y_pred = self.model.predict(self.X_test, verbose=0)
        return {
            "Features": self.features,
            "Target feature": self.target_feature,
            "MSE": metrics.mean_squared_error(y_true=self.y_test, y_pred=y_pred),
            "MAE": metrics.mean_absolute_error(y_true=self.y_test, y_pred=y_pred),
            "5% of mean Adj Close": self.data["Adj Close"].mean() * 0.05,
            "5% of mean Adj Close last 10 years": self.data["Adj Close"][
                self.data.index.date >= dt.date(self.data.index[-1].year - 10, 1, 1)
            ].mean()
            * 0.05,
        }

    def _save_trained_lstm_model(
        self, ticker_name: str, custom_filename: Optional[str]
    ) -> None:
        """Saves the trained LSTM model and logs its performance metrics.

        Args:
            ticker_name (str): The stock ticker name for saving the model.
        """
        if self.model is None:
            warnings.warn("The LSTM model wasn't yet created.")
            return

        if self.trained:
            log = self.get_model_statistics()

            if custom_filename is not None:
                self.model.save(f"{custom_filename}_model.keras")
                with open(f"{custom_filename}_log.json", "w") as file:
                    json.dump(log, file)
                    file.write("\n")
            else:
                self.model.save(
                    Path(__file__).parent.parent
                    / "data"
                    / "models"
                    / f"lstm_{ticker_name}.keras"
                )
                with open(
                    Path(__file__).parent.parent
                    / "data"
                    / "logs"
                    / f"log_{ticker_name}.json",
                    "w",
                ) as file:
                    json.dump(log, file)
                    file.write("\n")

    @classmethod
    def _load_trained_model(cls, data: pd.DataFrame, ticker_name: str) -> "LSTMModel":
        """Loads a previously trained LSTM model from file.

        Args:
            data (pd.DataFrame): The stock data.
            ticker_name (str): The stock ticker name.

        Returns:
            LSTMModel: The loaded model instance.
        """
        loaded_model = load_model(
            Path(__file__).parent.parent
            / "data"
            / "models"
            / f"lstm_{ticker_name}.keras"
        )
        with open(
            Path(__file__).parent.parent / "data" / "logs" / f"log_{ticker_name}.json",
            "r",
        ) as file:
            logs = json.load(file)

        lstm_model = LSTMModel(
            data=data, features=logs["Features"], target_feature=logs["Target feature"]
        )
        lstm_model.model = loaded_model
        lstm_model.trained = True
        lstm_model.split_scale_sequence()

        return lstm_model

import yfinance as yf
from joblib import load
import pandas as pd
from typing import Tuple, Any
from src.features.indicators import calculate_all_indicators
from src.config.settings import settings

# --- Custom Exceptions ---
class DataFetchError(Exception):
    """Custom exception for yfinance data fetching errors."""
    pass

class ScalerLoadError(Exception):
    """Custom exception for errors loading the scaler file."""
    pass

# --- Service Class ---
class DataService:
    def __init__(self, scaler_path: str):
        try:
            # Load the scaler object once during initialization
            self.scaler = load(scaler_path)
        except FileNotFoundError:
            raise ScalerLoadError(f"Scaler file not found at {scaler_path}")
        except Exception as e:
            raise ScalerLoadError(f"Error loading scaler: {e}")

    def _fetch_btc_data(self) -> pd.DataFrame:
        """
        Fetches the last 200 days of Bitcoin price data.
        """

        try:
            btc_ticker = yf.Ticker("BTC-USD")
            btc_historical_data = btc_ticker.history(period = "200d", actions = False)
            if btc_historical_data.empty:
                raise DataFetchError("yfinance returned an empty DataFrame.")
            return btc_historical_data
        except Exception as e:
            raise DataFetchError(f"Failed to fetch data from yfinance: {e}")

    def get_processed_features(self) -> Tuple[Any, float]:
        """
        Orchestrates data fetching, feature engineering, and preprocessing.
        
        Returns:
            A tuple containing:
            - The scaled feature array for prediction.
            - The last known price.
        """

        # 1. Fetch raw data
        historical_data = self._fetch_btc_data()

        # 2. Remove timezone localization
        historical_data = historical_data.tz_localize(None)

        # 3. Calculate financial indicators
        historical_data = calculate_all_indicators(historical_data)

        # 4. Sort data by index in descending order
        historical_data = historical_data.sort_index(ascending=False)

        # 5. Get the last known price (from 'Close' column, index 3)
        last_price = historical_data.iloc[0, 3]

        # 6. Select the first row (most recent data) for features
        input_data = historical_data.iloc[0, :]

        # 7. Replace missing values with 0
        input_data = input_data.fillna(0)

        # 8. Convert to array
        input_data_array = input_data.array

        # 9. Reshape for the model
        input_data_reshaped = input_data_array.reshape(1, -1)

        # 10. Standardize the input data
        scaled_data = self.scaler.transform(input_data_reshaped)

        return scaled_data, last_price

# --- Singleton Instance ---
try:
    data_service = DataService(scaler_path=settings.SCALER_FILE_PATH)
except ScalerLoadError as e:
    print(f"CRITICAL: Failed to load scaler. Error: {e}")
    data_service = None
from joblib import load
from typing import Dict, Any
from src.services.data_service import data_service, DataFetchError
from src.models.schemas import PredictionRequest
from src.config.settings import settings

# --- Custom Exceptions ---
class ModelLoadError(Exception):
    """Custom exception for errors loading the model file."""
    pass

# --- Service Class ---
class PredictionService:
    def __init__(self, model_path: str):
        try:
            self.model = load(model_path)
        except FileNotFoundError:
            raise ModelLoadError(f"Model file not found at {model_path}")
        except Exception as e:
            raise ModelLoadError(f"Error loading model: {e}")

    def create_prediction(self, request_data: PredictionRequest) -> Dict[str, Any]:
        """
        Generates a price prediction.
        """
        
        if data_service is None:
            raise DataFetchError("DataService is not available.")
            
        scaled_features, last_price = data_service.get_processed_features()
        prediction_array = self.model.predict(scaled_features)
        prediction_value = prediction_array.tolist()[0]

        response = {
            "Model": request_data.Model,
            "Last_Price": round(last_price, 2),
            "Prediction_For_Next_Day": round(prediction_value, 2)
        }
        return response

# --- Singleton Instance ---
try:
    prediction_service = PredictionService(model_path=settings.MODEL_FILE_PATH)
except ModelLoadError as e:
    print(f"CRITICAL: Failed to load model. Error: {e}")
    prediction_service = None
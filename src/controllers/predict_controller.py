from src.models.schemas import PredictionRequest
from src.services.prediction_service import prediction_service, ModelLoadError
from src.services.data_service import DataFetchError

def handle_prediction_request(request_body: PredictionRequest):
    """
    Handles the prediction request by calling the prediction service.
    
    Exceptions from the service (ModelLoadError, DataFetchError)
    are expected to be raised and caught by the main exception handlers.
    """
    
    # Check if service is available
    if prediction_service is None:
        raise ModelLoadError("Prediction service is not available.")
        
    response_data = prediction_service.create_prediction(request_body)
    
    return response_data
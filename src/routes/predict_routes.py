from fastapi import APIRouter
from src.models.schemas import PredictionRequest, PredictionResponse
from src.controllers import predict_controller

# Create a new router
router = APIRouter()

@router.post(
    "/predict", 
    response_model = PredictionResponse, 
    tags = ["Prediction"]
)
async def predict_price(request_body: PredictionRequest):
    """
    Endpoint for predicting the next day's Bitcoin price.
    
    This endpoint calculates all features on-the-fly.
    """

    # The route calls the controller to handle the logic
    return predict_controller.handle_prediction_request(request_body)
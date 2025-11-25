from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """
    Input schema for the /predict endpoint.
    """

    Model: str = Field(..., example="Machine Learning")

class PredictionResponse(BaseModel):
    """
    Output schema for the /predict endpoint.
    """

    Model: str = Field(..., example = "Machine Learning")
    Last_Price: float = Field(..., example = 65000.50)
    Prediction_For_Next_Day: float = Field(..., example = 65100.75)
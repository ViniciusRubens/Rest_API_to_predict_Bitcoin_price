# src/main.py
# Main application entry point

import uvicorn # Create WSGI server
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from src.routes import predict_routes
from src.services.data_service import DataFetchError, ScalerLoadError, data_service
from src.services.prediction_service import ModelLoadError, prediction_service

# --- Lifespan Event Handler (Novo) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    This replaces the deprecated on_event("startup").
    """
    # --- Startup Logic ---
    print("--- Application Startup ---")
    if prediction_service is None or data_service is None:
        print("CRITICAL ERROR: Model or Scaler failed to load. Check logs.")
    else:
        print("Application startup: Model and Scaler loaded successfully.")
    
    yield
    print("--- Application Shutdown ---")

# --- FastAPI App Instance ---

# Define metadata for API documentation
tags_metadata = [
    {"name": "Prediction", "description": "Endpoints for price prediction"}
]

# Create the FastAPI app instance and assign the lifespan handler
app = FastAPI(
    title = "Bitcoin Price API",
    description = "Project 2",
    version = "1.0",
    openapi_tags = tags_metadata,
    lifespan = lifespan
)

# --- Exception Handlers (Global) ---
# These catch errors from any part of the app

@app.exception_handler(DataFetchError)
async def data_fetch_exception_handler(request: Request, exc: DataFetchError):
    return JSONResponse(
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
        content = {"detail": f"Error fetching external data: {exc}"}
    )

@app.exception_handler(ModelLoadError)
@app.exception_handler(ScalerLoadError)
async def critical_load_exception_handler(request: Request, exc: (ModelLoadError, ScalerLoadError)):
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content = {"detail": f"Internal configuration error: {exc}"}
    )

# --- API Endpoints ---

@app.get("/")
def read_root():
    """
    Root endpoint to check if the API is running.
    """

    return {"message": "Bitcoin Price Prediction API. Use the /predict POST endpoint."}


app.include_router(predict_routes.router)

# Main entry point to run the server
if __name__ == "__main__":
    # Note: uvicorn path is now "src.main:app"
    uvicorn.run("src.main:app", host="0.0.0.0", port=3000, reload=True)
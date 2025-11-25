from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Manages application settings loaded from environment variables (.env file).
    """

    MODEL_FILE_PATH: str = "/data/model.joblib"
    SCALER_FILE_PATH: str = "/data/scaler.bin"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

# Create a settings instance
settings = Settings()
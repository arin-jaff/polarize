from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "polarize"

    # JWT Auth
    secret_key: str = "dev-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    # Garmin API (OAuth 1.0a)
    garmin_consumer_key: Optional[str] = None
    garmin_consumer_secret: Optional[str] = None

    # Concept2 API (OAuth 2.0)
    concept2_client_id: Optional[str] = None
    concept2_client_secret: Optional[str] = None
    concept2_redirect_uri: str = "http://localhost:8000/api/v1/integrations/concept2/callback"

    # Ollama AI
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_name: str = "fitness-coach"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

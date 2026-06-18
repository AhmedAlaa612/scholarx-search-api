from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    # Qdrant
    qdrant_api_key: str
    qdrant_endpoint: str
    qdrant_collection: str = "opportunities_v1"

    # PostgreSQL
    db_host: str
    db_port: int = 6543
    db_name: str = "postgres"
    db_user: str
    db_password: str

    # Jina
    jina_api_key: str
    jina_endpoint: str = "https://api.jina.ai/v1/embeddings"
    jina_model: str = "jina-embeddings-v3"

    # LLM (match existing .env key names)
    grok_api: str  # Groq API key (named grok_api in .env)
    cerebras_api: str  # Cerebras API key

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = '[*]'

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.cors_origins)

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration loaded from environment variables/.env.

    Keeps all runtime settings in one place so services/tools can import `settings`
    instead of reading env vars directly.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider switch: "ollama" for local models, "openai" for OpenAI-compatible APIs.
    llm_provider: str = "ollama"  # ollama | openai

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "tcs_forecast"

    data_dir: str = "./data"
    chroma_dir: str = "./data/chroma"
    log_level: str = "INFO"

    @property
    def mysql_url(self) -> str:
        """Build SQLAlchemy MySQL URL.

        Credentials are URL-encoded so special chars like '@', ':', '/' in passwords
        do not break the connection string.
        """
        from urllib.parse import quote_plus

        user = quote_plus(self.mysql_user)
        password = quote_plus(self.mysql_password)
        return (
            f"mysql+pymysql://{user}:{password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
        )

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def reports_path(self) -> Path:
        return self.data_path / "reports"

    @property
    def transcripts_path(self) -> Path:
        return self.data_path / "transcripts"

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)

    @property
    def llm_configured(self) -> bool:
        """Quick readiness check used by /health and request validation."""
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        return bool(self.ollama_base_url and self.ollama_model)


settings = Settings()

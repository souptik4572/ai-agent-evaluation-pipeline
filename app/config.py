from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./eval_pipeline.db"

    @field_validator("database_url")
    @classmethod
    def fix_async_driver(cls, v: str) -> str:
        # Render (and most hosted providers) give a plain postgresql:// URL.
        # SQLAlchemy async requires postgresql+asyncpg://.
        # Normalise here so neither the Render env var nor .env needs manual editing.
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        if v.startswith("postgres://"):
            # Some providers (Heroku, Supabase) use the older postgres:// alias
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        return v
    openai_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    log_level: str = "INFO"
    latency_threshold_ms: int = 1000
    annotator_agreement_threshold: float = 0.7
    confidence_auto_label_threshold: float = 0.85
    confidence_human_review_threshold: float = 0.6
    regression_check_eval_threshold: int = 5  # trigger auto regression check after N evals

    class Config:
        env_file = ".env"


settings = Settings()

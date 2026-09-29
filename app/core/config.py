"""Application settings, loaded from environment variables (and .env)."""
from __future__ import annotations

from datetime import date
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Databases: the owner role runs migrations, logging and approvals;
    # the read-only role is the only connection the SQL tool ever gets.
    database_url: str = "postgresql+psycopg://retailops:retailops@localhost:5432/retailops"
    database_url_ro: str = "postgresql+psycopg://retailops_ro:retailops_ro@localhost:5432/retailops"

    # LLM provider: anthropic (paid) or a free/local OpenAI-compatible one: gemini | groq | ollama.
    llm_provider: str = "anthropic"
    # Empty = the provider's default (claude-haiku-4-5, gemini-3.8-flash, openai/gpt-oss-120b, qwen2.5:7b).
    llm_model: str = ""
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    # Seconds between LLM requests; empty = provider default (Gemini free tier: 12.5 s = under 5/min).
    llm_min_interval_s: float | None = None
    llm_max_tokens: int = 4096
    usd_to_gbp: float = 0.79

    # Agent limits
    step_limit: int = 8
    sql_max_retries: int = 2
    sql_timeout_ms: int = 5000
    sql_row_limit: int = 1000
    # The dataset ends on 2011-12-09, so "today" for the agent is pinned to that date.
    as_of_date: date = date(2011, 12, 9)

    # Retrieval
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    retrieve_k: int = 20
    rerank_k: int = 5

    # API
    require_api_key: bool = True
    rate_limit: str = "20/minute"
    daily_spend_cap_gbp: float = 2.0
    response_cache_ttl_s: int = 3600
    log_level: str = "INFO"

    # Observability (optional)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Forecasting
    forecast_artifacts_dir: str = "forecasting/artifacts"


    @property
    def default_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        if self.llm_provider == "anthropic":
            return "claude-haiku-4-5"
        from agent.openai_compat import PROVIDERS

        return PROVIDERS[self.llm_provider]["default_model"]

    @property
    def llm_key_configured(self) -> bool:
        import os

        key = {"anthropic": self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"),
               "gemini": self.gemini_api_key, "groq": self.groq_api_key, "ollama": "local"}
        return bool(key.get(self.llm_provider))


@lru_cache
def get_settings() -> Settings:
    return Settings()

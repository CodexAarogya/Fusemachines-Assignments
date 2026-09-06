"""
Central configuration for the AI Assistant.

All tunables (model choice, generation parameters, RAG settings,
reliability knobs) live here and are sourced from environment
variables / .env so the same image can be re-deployed with different
behaviour without a code change.
"""
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Providers ---
    primary_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    local_llm_base_url: str = "http://vllm:8000/v1"
    local_llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    local_llm_api_key: str = "not-needed"

    provider_fallback_order: str = "openai,local,anthropic"

    # --- Generation parameters ---
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 1024

    # --- RAG ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_dir: str = "/data/chroma"
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 4

    # --- Reliability ---
    rate_limit_per_minute: int = 60
    max_concurrent_requests: int = 16
    cache_ttl_seconds: int = 600
    cache_max_size: int = 1000
    retry_max_attempts: int = 3
    retry_backoff_seconds: float = 1.0

    # --- App ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @property
    def fallback_order(self) -> List[str]:
        return [p.strip() for p in self.provider_fallback_order.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

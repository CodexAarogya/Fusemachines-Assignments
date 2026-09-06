"""
Pydantic models used for:
  1) FastAPI request/response validation
  2) Forcing/validating structured JSON output from the LLM
"""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# API request / response contracts
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    session_id: str = Field(default="default", description="Conversation/session id")
    use_rag: bool = Field(default=True, description="Whether to retrieve context from the vector store")
    structured: bool = Field(default=False, description="Whether to force structured JSON output")
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float


class ToolCallRecord(BaseModel):
    name: str
    arguments: Dict[str, Any]
    result: Any


class ChatResponse(BaseModel):
    answer: str
    structured_output: Optional[Dict[str, Any]] = None
    citations: List[Citation] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    provider_used: str
    cached: bool = False
    latency_ms: float


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_created: int


class HealthResponse(BaseModel):
    status: str
    providers: Dict[str, bool]


# ---------------------------------------------------------------------------
# Example structured-output schema the model is asked to fill in.
# This demonstrates "guaranteed valid JSON" for Task 1's structured output
# requirement. Swap/extend this for your own domain.
# ---------------------------------------------------------------------------
class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class QueryAnalysis(BaseModel):
    """Structured extraction the model must produce for `structured=True` requests."""
    intent: str = Field(..., description="Short label for what the user wants")
    sentiment: Sentiment
    entities: List[str] = Field(default_factory=list)
    requires_followup: bool
    summary: str = Field(..., description="One-sentence summary of the answer")


QUERY_ANALYSIS_JSON_SCHEMA = QueryAnalysis.model_json_schema()

"""
Pydantic models used for:
  1) FastAPI request/response validation
  2) Forcing/validating structured JSON output from the LLM (Task 1)
  3) The agentic loop's step-by-step decision schema (Task 3)
"""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# API request / response contracts (Task 1 / Task 2)
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
    total_tokens: int = 0


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_created: int


class HealthResponse(BaseModel):
    status: str
    providers: Dict[str, bool]


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


# ---------------------------------------------------------------------------
# Agentic loop contracts (Task 3)
# ---------------------------------------------------------------------------
class AgentAction(str, Enum):
    rag_search = "rag_search"
    wikipedia_search = "wikipedia_search"
    calculator = "calculator"
    get_weather = "get_weather"
    draft_answer = "draft_answer"
    ask_clarification = "ask_clarification"
    finish = "finish"


class AgentStep(BaseModel):
    """The structured decision the model must emit on every loop iteration.
    This IS the agentic decision point: the model chooses `action` based on
    everything gathered so far, not a position in a fixed pipeline."""
    thought: str = Field(..., description="Brief reasoning about what to do next and why")
    action: AgentAction
    action_input: Dict[str, Any] = Field(default_factory=dict)
    draft_answer: Optional[str] = Field(default=None, description="Set when action is draft_answer or finish")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


AGENT_STEP_JSON_SCHEMA = AgentStep.model_json_schema()


class EvidenceNote(BaseModel):
    """A compact, structured record of one piece of evidence. This is the
    'structured external notes' context-engineering artifact: raw tool
    output is discarded after this note is extracted from it."""
    iteration: int
    source: str
    content: str
    supports_answer: Optional[bool] = None


class StopReason(str, Enum):
    finished = "finished"
    ask_clarification = "ask_clarification"
    max_iterations = "max_iterations"
    loop_detected = "loop_detected"
    hard_failure = "hard_failure"


class AgentRequest(BaseModel):
    query: str
    session_id: str = "default"
    # If resuming a session that previously asked for clarification:
    clarification_answer: Optional[str] = None


class AgentTrace(BaseModel):
    iteration: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    tokens_used: int = 0
    raw_result: Optional[Any] = None  # raw tool output, kept ONLY in the trace (never in the prompt)


class AgentResponse(BaseModel):
    session_id: str
    final_answer: Optional[str] = None
    clarification_question: Optional[str] = None
    stop_reason: StopReason
    confidence: float = 0.0
    sources_used: List[str] = Field(default_factory=list)
    notes: List[EvidenceNote] = Field(default_factory=list)
    trace: List[AgentTrace] = Field(default_factory=list)
    iterations_used: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0

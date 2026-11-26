"""Pydantic request/response schemas for the FastAPI layer.

These models define the external API contract and validate agent output before
it is returned or logged.
"""

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    query: str = Field(
        default="Analyze the financial reports and transcripts for the last three quarters and provide a qualitative forecast for the upcoming quarter.",
        description="Analytical task for the agent",
    )
    quarters: int = Field(default=3, ge=1, le=4)


class FinancialMetric(BaseModel):
    metric: str
    value: str
    period: str
    source: str
    notes: str | None = None


class FinancialTrend(BaseModel):
    trend: str
    evidence: str
    direction: Literal["positive", "negative", "neutral", "mixed"]


class RiskOpportunity(BaseModel):
    type: Literal["risk", "opportunity"]
    title: str
    detail: str
    source: str


class ForecastResponse(BaseModel):
    company: str = "Tata Consultancy Services (TCS)"
    generated_at: datetime
    query: str
    quarters_analyzed: list[str]
    key_financial_metrics: list[FinancialMetric]
    financial_trends: list[FinancialTrend]
    management_outlook: str
    risks_and_opportunities: list[RiskOpportunity]
    qualitative_forecast: str
    confidence: Literal["low", "medium", "high"]
    sources_used: list[str]
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    market_context: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    database: str
    llm_configured: bool
    documents_available: dict[str, int]

"""Basic smoke test without calling external LLM. Checks imports and schema validation."""
from datetime import datetime
from app.models.schemas import ForecastResponse

sample = {
    "company": "Tata Consultancy Services (TCS)",
    "generated_at": datetime.utcnow(),
    "query": "test",
    "quarters_analyzed": ["Q3 FY2025-26"],
    "key_financial_metrics": [{"metric": "Revenue", "value": "100", "period": "Q3", "source": "test.pdf", "notes": None}],
    "financial_trends": [{"trend": "Stable growth", "evidence": "test", "direction": "positive"}],
    "management_outlook": "Cautiously optimistic",
    "risks_and_opportunities": [{"type": "risk", "title": "Margin pressure", "detail": "test", "source": "test.pdf"}],
    "qualitative_forecast": "Stable outlook",
    "confidence": "medium",
    "sources_used": ["test.pdf"],
    "tool_trace": [{"tool": "financial_data_extractor", "status": "ok", "summary": "test"}],
    "market_context": None,
}

resp = ForecastResponse.model_validate(sample)
print(resp.model_dump_json(indent=2)[:500])
print("SMOKE_OK")

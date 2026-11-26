"""Agent orchestration for the /forecast endpoint.

This module intentionally keeps orchestration simple and explicit:
1. Run deterministic/RAG tools to gather evidence.
2. Ask the LLM to synthesize that evidence into a strict JSON forecast.
3. Validate/normalize output into the ForecastResponse schema.

Keeping tool execution outside the final prompt generation makes behavior easier
to debug and reduces hallucination risk.
"""

import json
import logging
from datetime import datetime
from typing import Any

from app.config import settings
from app.models.schemas import ForecastResponse
from app.services.document_loader import Document, get_recent_quarters
from app.services.llm import get_chat_model
from app.tools.financial_data_extractor import FinancialDataExtractorTool
from app.tools.market_data_tool import MarketDataTool
from app.tools.qualitative_analysis import QualitativeAnalysisTool

logger = logging.getLogger(__name__)

# Master prompt used for final synthesis. Tool outputs are injected below as JSON.
# The schema is included explicitly because local/smaller models often drift without it.
MASTER_PROMPT = """You are a financial forecasting agent for Tata Consultancy Services (TCS).
Your job is to produce a grounded, evidence-based qualitative forecast for the upcoming quarter.

Rules:
- Use only the provided tool outputs and document evidence. Do not invent metrics or quotes.
- If a metric is unavailable, say "Not available in provided documents" in notes or omit it.
- Synthesize across quarters: identify trends, not just isolated facts.
- Be explicit about uncertainty and confidence.
- Return STRICT JSON matching the schema exactly. No markdown fences, no extra keys.

Required JSON schema:
{{
  "company": "Tata Consultancy Services (TCS)",
  "generated_at": "<ISO-8601>",
  "query": "<user query>",
  "quarters_analyzed": ["..."],
  "key_financial_metrics": [{{"metric":"","value":"","period":"","source":"","notes":""}}],
  "financial_trends": [{{"trend":"","evidence":"","direction":"positive|negative|neutral|mixed"}}],
  "management_outlook": "...",
  "risks_and_opportunities": [{{"type":"risk|opportunity","title":"","detail":"","source":""}}],
  "qualitative_forecast": "...",
  "confidence": "low|medium|high",
  "sources_used": ["..."],
  "tool_trace": [{{"tool":"","status":"","summary":""}}],
  "market_context": null
}}

User query: {query}
Quarters to analyze: {quarters}
Financial extraction tool output:
{financial_data}
Qualitative analysis tool output:
{qualitative_data}
Market data tool output (optional):
{market_data}
"""


class ForecastAgent:
    """Coordinates tools and LLM synthesis for a single forecast request."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        # Tools are constructed once with the loaded document corpus.
        self.financial_tool = FinancialDataExtractorTool(documents)
        self.qualitative_tool = QualitativeAnalysisTool(documents)
        self.market_tool = MarketDataTool()

    def _llm(self):
        """Create chat model on demand; provider is selected from settings."""
        if not settings.llm_configured:
            raise ValueError(
                "LLM not configured. Set LLM_PROVIDER=ollama (default) with Ollama running, or LLM_PROVIDER=openai with OPENAI_API_KEY in .env."
            )
        # Low temperature to keep JSON/forecast output more stable and grounded.
        return get_chat_model(temperature=0.2)

    def run(self, query: str, quarters: int = 3) -> ForecastResponse:
        """Execute tool pipeline, synthesize response, and return validated schema."""
        recent_quarters = get_recent_quarters(self.documents, n=quarters)
        tool_trace: list[dict[str, Any]] = []  # Returned to client/DB for observability.

        # Tool 1: financial extraction (deterministic parser + optional LLM enrichment + cache).
        try:
            financial_data = self.financial_tool.run(recent_quarters)
            tool_trace.append({"tool": self.financial_tool.name, "status": "ok", "summary": f"Extracted metrics for {len(financial_data.get('results', []))} report(s)"})
        except Exception as e:
            logger.exception("Financial extraction failed")
            financial_data = {"error": str(e)}
            tool_trace.append({"tool": self.financial_tool.name, "status": "error", "summary": str(e)})

        # Tool 2: qualitative RAG over transcripts, or report fallback when transcripts missing.
        try:
            qualitative_data = self.qualitative_tool.run(query, recent_quarters)
            summary = "Analyzed transcript themes/sentiment/risks"
            if isinstance(qualitative_data, dict) and qualitative_data.get("using_fallback_reports"):
                summary = "Analyzed report documents (no transcripts found) for themes/sentiment/risks"
            tool_trace.append({"tool": self.qualitative_tool.name, "status": "ok", "summary": summary})
        except Exception as e:
            logger.exception("Qualitative analysis failed")
            qualitative_data = {"error": str(e)}
            tool_trace.append({"tool": self.qualitative_tool.name, "status": "error", "summary": str(e)})

        # Tool 3: optional market context. Failures are non-fatal by design.
        try:
            market_data = self.market_tool.run()
            status = "ok" if "error" not in market_data else "warning"
            tool_trace.append({"tool": self.market_tool.name, "status": status, "summary": "Fetched live market snapshot"})
        except Exception as e:
            logger.warning("Market data failed: %s", e)
            market_data = {"error": str(e)}
            tool_trace.append({"tool": self.market_tool.name, "status": "warning", "summary": str(e)})

        # Cap payload sizes to keep prompt within local-model context limits.
        prompt = MASTER_PROMPT.format(
            query=query,
            quarters=recent_quarters,
            financial_data=json.dumps(financial_data, ensure_ascii=False)[:18000],
            qualitative_data=json.dumps(qualitative_data, ensure_ascii=False)[:18000],
            market_data=json.dumps(market_data, ensure_ascii=False),
        )

        llm = self._llm()
        resp = llm.invoke(prompt)
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        raw = raw.strip()
        # Some models wrap JSON in markdown fences despite instructions.
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = self._parse_json(raw)
        if data is None:
            # One strict retry gives smaller/local models a chance to recover formatting.
            try:
                retry = llm.invoke("Return ONLY valid JSON for the previous response. No markdown, no commentary.\n\n" + raw[:12000])
                raw2 = retry.content if isinstance(retry.content, str) else str(retry.content)
                data = self._parse_json(raw2)
            except Exception:
                data = None
        if data is None:
            logger.error("Agent output JSON parse failed")
            data = self._fallback_response(query, recent_quarters, financial_data, qualitative_data, market_data, tool_trace, "Invalid JSON from LLM")

        # Normalize required fields so Pydantic validation is resilient to partial LLM output.
        data.setdefault("company", "Tata Consultancy Services (TCS)")
        data.setdefault("generated_at", datetime.utcnow().isoformat())
        data.setdefault("query", query)
        data.setdefault("quarters_analyzed", recent_quarters)
        data.setdefault("key_financial_metrics", [])
        data.setdefault("financial_trends", [])
        data.setdefault("management_outlook", "Not available in provided documents.")
        data.setdefault("risks_and_opportunities", [])
        data.setdefault("qualitative_forecast", "Insufficient grounded data to produce forecast.")
        data.setdefault("confidence", "low")
        data.setdefault("sources_used", self._source_names())
        data.setdefault("tool_trace", tool_trace)
        data.setdefault("market_context", market_data if isinstance(market_data, dict) and "error" not in market_data else None)
        # Use server-side timestamp/tool trace as source of truth.
        data["generated_at"] = datetime.utcnow()
        data["tool_trace"] = tool_trace
        data = self._normalize_response(data)

        return ForecastResponse.model_validate(data)

    def _parse_json(self, raw: str) -> dict | None:
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        # try direct parse
        try:
            return json.loads(raw)
        except Exception:
            pass
        # try extracting first JSON object
        import re
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

    def _source_names(self) -> list[str]:
        return [d.path.name for d in self.documents]

    def _normalize_response(self, data: dict) -> dict:
        """Coerce common LLM schema drift into the strict ForecastResponse shape."""
        if not isinstance(data, dict):
            return {}
        # financial_trends: ensure direction is present/valid
        norm_trends = []
        for t in data.get("financial_trends") or []:
            if not isinstance(t, dict):
                continue
            direction = str(t.get("direction", "neutral")).lower().strip()
            if direction not in {"positive", "negative", "neutral", "mixed"}:
                direction = "neutral"
            norm_trends.append({
                "trend": str(t.get("trend", "Not available")),
                "evidence": str(t.get("evidence", "Not available in provided documents")),
                "direction": direction,
            })
        data["financial_trends"] = norm_trends

        # risks/opportunities: ensure type/title/detail/source
        norm_ro = []
        for r in data.get("risks_and_opportunities") or []:
            if not isinstance(r, dict):
                continue
            rtype = str(r.get("type", "risk")).lower().strip()
            if rtype not in {"risk", "opportunity"}:
                rtype = "risk"
            norm_ro.append({
                "type": rtype,
                "title": str(r.get("title") or r.get("name") or "Not specified"),
                "detail": str(r.get("detail") or r.get("description") or "Not available in provided documents"),
                "source": str(r.get("source") or "Not available in provided documents"),
            })
        data["risks_and_opportunities"] = norm_ro

        conf = str(data.get("confidence", "low")).lower().strip()
        data["confidence"] = conf if conf in {"low", "medium", "high"} else "low"
        return data

    def _fallback_response(self, query, quarters, financial_data, qualitative_data, market_data, tool_trace, err):
        return {
            "company": "Tata Consultancy Services (TCS)",
            "generated_at": datetime.utcnow().isoformat(),
            "query": query,
            "quarters_analyzed": quarters,
            "key_financial_metrics": [],
            "financial_trends": [],
            "management_outlook": "Model output could not be parsed into required schema.",
            "risks_and_opportunities": [],
            "qualitative_forecast": f"Fallback response due to parse error: {err}",
            "confidence": "low",
            "sources_used": self._source_names(),
            "tool_trace": tool_trace + [{"tool": "llm_synthesis", "status": "error", "summary": err}],
            "market_context": market_data if isinstance(market_data, dict) and "error" not in market_data else None,
        }

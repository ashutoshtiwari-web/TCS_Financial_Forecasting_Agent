"""FinancialDataExtractorTool: extracts structured metrics from report PDFs.

Design: deterministic parsing first (fast/reliable for tables), optional LLM
enrichment second (captures extra metrics), and per-PDF cache to avoid repeated work.
"""

import json
import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.services.document_loader import Document
from app.services.financial_parser import parse_financial_report
from app.services.llm import get_chat_model

logger = logging.getLogger(__name__)


class ExtractedMetric(BaseModel):
    metric: str
    value: str
    period: str
    source: str
    notes: str | None = None


class FinancialExtractionResult(BaseModel):
    quarter: str
    metrics: list[ExtractedMetric] = Field(default_factory=list)
    summary: str = ""


EXTRACTION_PROMPT = """You are a financial analyst extracting key metrics from a TCS quarterly financial report.
Extract metrics ONLY if supported by the document text. Do not invent numbers.
Return strict JSON with this schema:
{{
  "quarter": "...",
  "metrics": [
    {{"metric":"Total Revenue","value":"...","period":"...","source":"<filename>","notes":"..."}}
  ],
  "summary":"2-3 sentence financial summary grounded in the document"
}}

Required metrics to look for (include only those found): Total Revenue, Revenue Growth YoY/CC, Net Profit, Net Margin, Operating Margin, EBIT Margin, TCV/Order Book, Free Cash Flow, EPS, Headcount, Attrition, Dividend.
Quarter: {quarter}
Source file: {source}
Document text (truncated):
{content}
"""


def _get_llm():
    return get_chat_model(temperature=0)


def extract_metrics_from_report(doc: Document) -> FinancialExtractionResult:
    """Extract metrics for one report, preferring deterministic values."""
    # 1) Deterministic parsing for reliability on financial tables.
    parsed_metrics, parsed_summary = parse_financial_report(doc.path, doc.quarter)
    metrics = [ExtractedMetric(**m.__dict__) for m in parsed_metrics]
    summary = parsed_summary

    # 2) Optional LLM enrichment for metrics the regex parser didn't capture.
    #    We never overwrite deterministic metrics; LLM only fills gaps.
    try:
        llm = _get_llm()
        content = doc.text[:18000]
        prompt = EXTRACTION_PROMPT.format(quarter=doc.quarter, source=doc.path.name, content=content)
        resp = llm.invoke(prompt)
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        from app.services.json_utils import parse_json_loose
        data = parse_json_loose(raw) or {}
        existing = {m.metric.lower().strip() for m in metrics}
        for m in (data.get("metrics") or []):
            if not isinstance(m, dict):
                continue
            name = str(m.get("metric", "")).strip()
            if not name or name.lower() in existing:
                continue
            val = str(m.get("value", "")).strip()
            if not val or val.lower() in {"not available", "n/a", "na", "not specified", "not available in provided documents"}:
                continue
            metrics.append(ExtractedMetric(
                metric=name,
                value=val,
                period=str(m.get("period") or doc.quarter),
                source=str(m.get("source") or doc.path.name),
                notes=str(m.get("notes")) if m.get("notes") else "LLM-extracted; verify in source",
            ))
            existing.add(name.lower())
        if not summary and isinstance(data.get("summary"), str):
            summary = data["summary"]
    except Exception as e:
        # Enrichment is best-effort; deterministic extraction can still be useful.
        logger.warning("LLM enrichment failed for %s: %s", doc.path.name, e)

    if not metrics:
        return FinancialExtractionResult(quarter=doc.quarter, metrics=[], summary=summary or "No metrics extracted")
    return FinancialExtractionResult(quarter=doc.quarter, metrics=metrics, summary=summary)


class FinancialDataExtractorTool:
    """Extracts key financial metrics from quarterly report PDFs."""

    name = "financial_data_extractor"
    description = (
        "Extract key financial metrics (revenue, net profit, margins, TCV, etc.) "
        "from TCS quarterly financial report documents for specified quarters."
    )

    def __init__(self, documents: list[Document]):
        self.documents = [d for d in documents if d.doc_type == "report"]

    def run(self, quarters: list[str] | None = None) -> dict[str, Any]:
        """Extract metrics for requested quarters, using cache when possible."""
        from app.services.financial_cache import load_cached_extraction, save_cached_extraction

        # Filter loaded report docs by requested quarter labels.
        target_docs = self.documents
        if quarters:
            qset = {q.lower() for q in quarters}
            target_docs = [d for d in self.documents if d.quarter.lower() in qset or any(q in d.quarter.lower() for q in qset)]
        # Defensive fallback: if labels don't match, use first few available reports.
        if not target_docs and self.documents:
            target_docs = self.documents[:3]

        results = []
        cache_hits = 0
        cache_misses = 0
        for doc in target_docs:
            # Cache key includes content SHA256, so edited/replaced PDFs re-extract.
            cached = load_cached_extraction(doc.path)
            if cached is not None:
                cached["cache_hit"] = True
                results.append(cached)
                cache_hits += 1
                continue

            logger.info("Extracting metrics from %s", doc.path.name)
            res = extract_metrics_from_report(doc)
            result_dict = res.model_dump()
            result_dict["cache_hit"] = False
            save_cached_extraction(doc.path, result_dict)
            results.append(result_dict)
            cache_misses += 1
        return {
            "tool": self.name,
            "quarters_requested": quarters,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "results": results,
        }

    def as_langchain_tool(self):
        parent = self

        @tool("financial_data_extractor")
        def _tool(quarters_csv: str = "") -> str:
            """Extract key financial metrics from TCS quarterly financial reports. Input: comma-separated quarter labels like 'Q3 FY2025-26,Q2 FY2025-26' or empty for recent available reports."""
            quarters = [q.strip() for q in quarters_csv.split(",") if q.strip()] or None
            return json.dumps(parent.run(quarters), ensure_ascii=False)

        return _tool

"""Deterministic financial metric extraction from TCS report PDFs.

LLM-only extraction can be slow and inconsistent on numeric tables. This parser
uses regex/table-text heuristics to pull core metrics first; the LLM is then used
only for enrichment in FinancialDataExtractorTool.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass
class ParsedMetric:
    """Normalized metric record used by both parser and LLM enrichment paths."""
    metric: str
    value: str
    period: str
    source: str
    notes: str | None = None


def _to_float(s: str | None):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("₹", "").replace("Rs.", "").replace("\n", " ")
    s = s.replace("(", "-").replace(")", "")
    s = re.sub(r"[^\d.\-]", "", s)
    if s in {"", "-", "."}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _fmt_amount(v: float | None, unit: str = "₹ crore") -> str:
    if v is None:
        return "Not available in provided documents"
    if abs(v - round(v)) < 0.01:
        return f"{int(round(v)):,} {unit}"
    return f"{v:,.2f} {unit}"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "Not available in provided documents"
    return f"{v:.2f}%"


def _extract_text(path: Path) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        return ""


def _find_row_values(text: str, labels: list[str], n: int = 6) -> list[float | None] | None:
    # Match label followed by n numeric tokens (allows commas, parentheses, decimals)
    num = r"(\(?-?\d[\d,]*(?:\.\d+)?\)?)"
    for label in labels:
        for nn in [n, 5, 4, 3, 2]:
            pat = rf"{re.escape(label)}\s+" + r"\s+".join([num] * nn)
            m = re.search(pat, text, re.I)
            if m:
                vals = [_to_float(m.group(i)) for i in range(1, nn + 1)]
                # pad to n
                while len(vals) < n:
                    vals.append(None)
                return vals
    return None


def _search_first(text: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, re.I | re.S)
        if m:
            val = m.group(1).strip()
            val = re.sub(r"\s+", " ", val)
            return val[:120]
    return None


def parse_financial_report(path: Path, quarter: str) -> tuple[list[ParsedMetric], str]:
    """Extract key metrics from a TCS quarterly PDF.

    Returns (metrics, summary). This is intentionally heuristic: it favors reliable
    core numbers over broad but noisy extraction.
    """
    text = _extract_text(path)
    source = path.name
    metrics: list[ParsedMetric] = []

    # Consolidated P&L row patterns usually expose 6 values:
    # [Q current, Q previous, Q YoY, 6M current, 6M YoY, Year]
    rev = _find_row_values(text, ["Revenue from operations"])
    total_income = _find_row_values(text, ["TOTAL INCOME"])
    pbt_before_exc = _find_row_values(text, ["PROFIT BEFORE EXCEPTIONAL ITEMS AND TAX", "PROFIT BEFORE EXCEPTIONAL ITEM AND TAX", "Profit before exceptional items and tax"])
    pbt = _find_row_values(text, ["PROFIT BEFORE TAX"])
    profit = _find_row_values(text, ["PROFIT FOR THE PERIOD", "PROFIT FOR THE YEAR / PERIOD", "PROFIT FOR THE YEAR", "Profit for the period", "Profit for the year"])
    tax = _find_row_values(text, ["TOTAL TAX EXPENSE", "Total tax expense"])

    # Determine period labels from headers
    period_curr = quarter
    period_yoy = None
    m_yoy = re.search(r"Three months ended.*?September 30,\s*(20\d{2}).*?September 30,\s*(20\d{2})", text, re.I | re.S)
    if m_yoy:
        # heuristic only; keep quarter arg as primary
        pass

    def add(metric, value, period=period_curr, notes=None):
        metrics.append(ParsedMetric(metric=metric, value=value, period=period, source=source, notes=notes))

    if rev:
        add("Revenue from operations", _fmt_amount(rev[0]), notes="Consolidated, quarter")
        if rev[2] and rev[2] != 0:
            yoy = ((rev[0] - rev[2]) / abs(rev[2])) * 100
            add("Revenue Growth YoY", _fmt_pct(yoy), notes="vs same quarter previous year, consolidated")
        if len(rev) > 1 and rev[1] and rev[1] != 0:
            qoq = ((rev[0] - rev[1]) / abs(rev[1])) * 100
            add("Revenue Growth QoQ", _fmt_pct(qoq), notes="vs previous quarter, consolidated")
    if total_income:
        add("Total Income", _fmt_amount(total_income[0]), notes="Consolidated, quarter")
    if profit:
        add("Net Profit (Profit for the period)", _fmt_amount(profit[0]), notes="Consolidated, quarter")
        if rev and rev[0]:
            add("Net Margin", _fmt_pct((profit[0] / rev[0]) * 100), notes="Profit for period / Revenue from operations")
        if profit[2] and profit[2] != 0:
            yoy = ((profit[0] - profit[2]) / abs(profit[2])) * 100
            add("Net Profit Growth YoY", _fmt_pct(yoy), notes="vs same quarter previous year, consolidated")
    if pbt_before_exc and rev and rev[0]:
        add("Operating Margin (PBT before exceptional items)", _fmt_pct((pbt_before_exc[0] / rev[0]) * 100), notes="PBT before exceptional items / Revenue from operations")
    if tax:
        add("Total Tax Expense", _fmt_amount(tax[0]), notes="Consolidated, quarter")

    # Common extra metrics via text search
    eps = _search_first(text, [r"Basic EPS\s*\(?₹?\)?\s*[:\-]?\s*([0-9,.]+)", r"Earnings per share\s*\(Basic\)\s*₹?\s*([0-9,.]+)"])
    if eps:
        add("EPS (Basic)", f"₹ {eps}", notes="As reported; verify period in source")
    tcv = _search_first(text, [r"Total Contract Value\s*\(TCV\)\s*[:\-]?\s*([^\n]{0,60})", r"TCV\s+of\s+([^\n]{0,60})", r"order book[^.\n]{0,20}TCV[^.\n]{0,30}?([$₹]?\s*[0-9,.]+\s*billion[^\n]{0,20})"])
    if tcv:
        add("TCV / Order Book", tcv, notes="Extracted from text; verify in source")
    headcount = _search_first(text, [r"headcount[^\n]{0,40}?([0-9,]{5,7})", r"([0-9,]{6})\s+employees"])
    if headcount:
        add("Headcount", headcount, notes="Extracted from text; verify in source")
    attrition = _search_first(text, [r"attrition[^\n]{0,30}?([0-9]+(?:\.[0-9]+)?%)", r"LTM IT Services attrition[^\n]{0,20}?([0-9]+(?:\.[0-9]+)?%)"])
    if attrition:
        add("Attrition", attrition, notes="Extracted from text; verify in source")
    dividend = _search_first(text, [r"interim dividend[^\n]{0,40}?₹\s*([0-9]+)", r"dividend[^\n]{0,30}?`?\s*([0-9]+)\s*per equity share"])
    if dividend:
        add("Dividend", f"₹ {dividend} per share", notes="As reported; verify type/period in source")

    # Summary
    parts = []
    if rev:
        parts.append(f"Revenue from operations was {_fmt_amount(rev[0])}")
    if profit:
        parts.append(f"profit for the period was {_fmt_amount(profit[0])}")
    if rev and rev[2] and rev[2] != 0:
        parts.append(f"revenue YoY {((rev[0]-rev[2])/abs(rev[2])*100):.2f}%")
    summary = ". ".join(parts) + "." if parts else "Key financial metrics extracted deterministically from report text/tables."
    return metrics, summary

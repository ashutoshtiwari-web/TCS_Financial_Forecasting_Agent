"""Document loading and quarter inference.

Loads PDFs from data/reports and data/transcripts into lightweight Document objects
used by extraction and RAG tools. Extraction prefers pdfplumber and falls back to pypdf.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """In-memory representation of a loaded source document."""
    path: Path
    doc_type: str  # "report" or "transcript"
    quarter: str    # inferred label like "Q1 FY2026-27"
    text: str


QUARTER_RE = re.compile(r"Q([1-4])\s*(?:FY)?\s*(20\d{2})(?:[-/](\d{2,4}))?", re.I)


def infer_quarter(name: str) -> str:
    m = QUARTER_RE.search(name)
    if m:
        q = m.group(1)
        y1 = m.group(2)
        y2 = m.group(3)
        if y2:
            y2 = y2 if len(y2) == 4 else f"20{y2}" if int(y2) < 50 else f"19{y2}"
            return f"Q{q} FY{y1}-{y2[-2:]}"
        return f"Q{q} FY{y1}"
    # fallback: find FY pattern
    fy = re.search(r"FY\s*(20\d{2})[-/](\d{2})", name, re.I)
    if fy:
        return f"FY{fy.group(1)}-{fy.group(2)}"
    return name


def _extract_with_pdfplumber(path: Path) -> str:
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
            # also try tables for financial reports
            try:
                tables = page.extract_tables() or []
                for table in tables:
                    for row in table:
                        if row:
                            cells = [str(c).strip() for c in row if c and str(c).strip()]
                            if cells:
                                parts.append(" | ".join(cells))
            except Exception:
                pass
    return "\n".join(parts)


def _extract_with_pypdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def extract_text(path: Path) -> str:
    try:
        text = _extract_with_pdfplumber(path)
        if len(text.strip()) > 200:
            return text
    except Exception as e:
        logger.warning("pdfplumber failed for %s: %s", path.name, e)
    try:
        text = _extract_with_pypdf(path)
        if text.strip():
            return text
    except Exception as e:
        logger.warning("pypdf failed for %s: %s", path.name, e)
    logger.error("Could not extract text from %s", path)
    return ""


def load_documents(reports_dir: Path, transcripts_dir: Path) -> list[Document]:
    """Load all PDFs from report/transcript folders into Document objects."""
    docs: list[Document] = []
    for doc_type, folder in [("report", reports_dir), ("transcript", transcripts_dir)]:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.pdf")):
            text = extract_text(path)
            if not text.strip():
                continue
            quarter = infer_quarter(path.stem)
            docs.append(Document(path=path, doc_type=doc_type, quarter=quarter, text=text))
            logger.info("Loaded %s %s (%d chars)", doc_type, path.name, len(text))
    return docs


def get_recent_quarters(docs: list[Document], n: int = 3) -> list[str]:
    """Pick most recent quarter labels by heuristic FY/Q parsing."""
    # Sort by quarter string heuristically: FY2026-27 Q3 > ...
    def q_key(q: str):
        m = re.search(r"Q([1-4]).*?FY\s*(20\d{2})[-/](\d{2})", q, re.I)
        if m:
            return (int(m.group(2)), int(m.group(3)), int(m.group(1)))
        m = re.search(r"FY\s*(20\d{2})[-/](\d{2})", q, re.I)
        if m:
            return (int(m.group(1)), int(m.group(2)), 0)
        return (0, 0, 0)

    quarters = sorted({d.quarter for d in docs}, key=q_key, reverse=True)
    return quarters[:n]

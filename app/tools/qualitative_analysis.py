"""QualitativeAnalysisTool: RAG over transcripts/reports for themes and outlook.

Uses Chroma for semantic retrieval, then an LLM to summarize evidence into a
structured JSON analysis. Falls back to report documents when transcripts are absent.
"""

import json
import logging
from typing import Any

from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from app.config import settings
from app.services.document_loader import Document
from app.services.llm import get_chat_model, get_embeddings

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are an equity research analyst analyzing TCS earnings call transcripts.
Using ONLY the provided transcript excerpts, identify:
1. Recurring themes across quarters
2. Management sentiment (optimistic/cautious/mixed) with evidence
3. Forward-looking statements/guidance
4. Significant risks or opportunities mentioned

Return strict JSON:
{{
  "recurring_themes": [{{"theme":"...","evidence":"...","quarters":["..."]}}],
  "management_sentiment": {{"overall":"optimistic|cautious|mixed","summary":"...","evidence":["..."]}},
  "forward_looking_statements": [{{"statement":"...","source_quarter":"..."}}],
  "risks": [{{"title":"...","detail":"...","source":"..."}}],
  "opportunities": [{{"title":"...","detail":"...","source":"..."}}]
}}

Question: {question}
Transcript excerpts:
{context}
"""


def _get_llm():
    return get_chat_model(temperature=0.1)


def _get_embeddings():
    return get_embeddings()


class QualitativeAnalysisTool:
    """RAG-based semantic analysis over earnings call transcripts."""

    name = "qualitative_analysis"
    description = (
        "Perform semantic search and qualitative analysis across TCS earnings call transcripts "
        "to identify recurring themes, management sentiment, forward-looking statements, risks, and opportunities."
    )

    def __init__(self, documents: list[Document]):
        # Prefer transcripts for management tone/outlook; fallback to reports if unavailable.
        self.transcript_docs = [d for d in documents if d.doc_type == "transcript"]
        self.report_docs = [d for d in documents if d.doc_type == "report"]
        self.documents = self.transcript_docs or self.report_docs
        self.using_fallback = not self.transcript_docs and bool(self.report_docs)
        self._vectorstore: Chroma | None = None  # Lazy-built on first use.

    def _build_vectorstore(self) -> Chroma | None:
        """Chunk documents and build an in-process Chroma index for retrieval."""
        if not self.documents:
            return None
        # Moderate chunks balance semantic coherence with retrieval granularity.
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        texts, metadatas = [], []
        for doc in self.documents:
            chunks = splitter.split_text(doc.text)
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append(
                    {
                        "source": doc.path.name,
                        "quarter": doc.quarter,
                        "doc_type": doc.doc_type,
                        "chunk": i,
                    }
                )
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        embeddings = _get_embeddings()
        # Reset stale/incompatible local Chroma collections to avoid schema/version issues.
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(settings.chroma_path))
            for name in ["tcs_qualitative", "tcs_transcripts"]:
                try:
                    client.delete_collection(name)
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Could not reset Chroma collection: %s", e)
        vs = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            collection_name="tcs_qualitative",
            persist_directory=str(settings.chroma_path),
        )
        logger.info("Built transcript vectorstore with %d chunks", len(texts))
        return vs

    @property
    def vectorstore(self) -> Chroma | None:
        if self._vectorstore is None:
            self._vectorstore = self._build_vectorstore()
        return self._vectorstore

    def run(self, question: str, quarters: list[str] | None = None, k: int = 8) -> dict[str, Any]:
        """Retrieve relevant excerpts and ask LLM for structured qualitative analysis."""
        if not self.documents:
            return {"tool": self.name, "warning": "No transcript/report documents available; qualitative analysis limited.", "analysis": {}}

        context_parts = []
        if self.vectorstore is not None:
            try:
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
                docs = retriever.invoke(question)
                for d in docs:
                    meta = d.metadata or {}
                    context_parts.append(
                        f"[Source: {meta.get('source','unknown')} | Quarter: {meta.get('quarter','unknown')}]\n{d.page_content}"
                    )
            except Exception as e:
                logger.warning("Vector retrieval failed, falling back to direct text: %s", e)

        # Fallback when retrieval fails or returns nothing: include recent doc excerpts directly.
        if not context_parts:
            selected = self.documents[:3]
            for doc in selected:
                context_parts.append(f"[Source: {doc.path.name} | Quarter: {doc.quarter}]\n{doc.text[:4000]}")

        context = "\n\n---\n\n".join(context_parts)[:20000]
        llm = _get_llm()
        prompt = ANALYSIS_PROMPT.format(question=question, context=context)
        resp = llm.invoke(prompt)
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        from app.services.json_utils import parse_json_loose
        analysis = parse_json_loose(raw)
        if analysis is None:
            retry = llm.invoke("Return ONLY valid JSON object. No markdown.\n\n" + prompt[:14000])
            raw2 = retry.content if isinstance(retry.content, str) else str(retry.content)
            analysis = parse_json_loose(raw2)
        if analysis is None:
            logger.error("Qualitative analysis parse failed: raw=%r", (raw or "")[:300])
            analysis = {"error": "Model did not return valid JSON", "raw": (raw or "")[:2000]}

        sources = []
        if 'docs' in locals():
            sources = list({(d.metadata or {}).get("source", "") for d in docs if (d.metadata or {}).get("source")})
        if not sources:
            sources = [d.path.name for d in self.documents[:3]]
        return {
            "tool": self.name,
            "question": question,
            "quarters_requested": quarters,
            "using_fallback_reports": self.using_fallback,
            "sources": sources,
            "analysis": analysis,
        }

    def as_langchain_tool(self):
        parent = self

        @tool("qualitative_analysis")
        def _tool(question: str) -> str:
            """Analyze TCS earnings call transcripts for themes, management sentiment, forward-looking statements, risks, and opportunities. Input: a specific analytical question."""
            return json.dumps(parent.run(question), ensure_ascii=False)

        return _tool

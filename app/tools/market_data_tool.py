"""Optional live market context tool.

This is intentionally non-blocking: forecast generation should still succeed when
yfinance/network data is unavailable.
"""

import json
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class MarketDataTool:
    """Fetches a lightweight TCS.NS market snapshot via yfinance."""

    name = "market_data"
    description = "Fetch live market data for TCS (stock price, recent performance) for additional context."

    def run(self) -> dict[str, Any]:
        """Return latest market snapshot, or an error payload on failure."""
        try:
            import yfinance as yf

            ticker = yf.Ticker("TCS.NS")
            info = ticker.fast_info
            hist = ticker.history(period="5d")
            last = hist.iloc[-1] if not hist.empty else None
            prev = hist.iloc[-2] if len(hist) > 1 else None
            change_pct = None
            if last is not None and prev is not None and prev["Close"]:
                change_pct = ((last["Close"] - prev["Close"]) / prev["Close"]) * 100
            return {
                "tool": self.name,
                "symbol": "TCS.NS",
                "last_price": float(info.last_price) if getattr(info, "last_price", None) else None,
                "previous_close": float(info.previous_close) if getattr(info, "previous_close", None) else None,
                "day_change_pct": round(float(change_pct), 2) if change_pct is not None else None,
                "currency": "INR",
                "source": "yfinance",
            }
        except Exception as e:
            # Do not raise: caller treats market data as optional context.
            logger.warning("Market data fetch failed: %s", e)
            return {"tool": self.name, "error": str(e), "symbol": "TCS.NS"}

    def as_langchain_tool(self):
        parent = self

        @tool("market_data")
        def _tool(_: str = "") -> str:
            """Fetch current TCS market data (stock price and recent change) for context."""
            return json.dumps(parent.run(), ensure_ascii=False)

        return _tool

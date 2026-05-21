"""
mcp_airbnb.py
--------------
Airbnb MCP Client implementation.
"""

import asyncio
import json
import logging
from typing import Any

from mcp import StdioServerParameters
from .base import BaseMCPClient

logger = logging.getLogger(__name__)

# ── MCP Server config ────────────────────────────────────────────────────
AIRBNB_SERVER = StdioServerParameters(
    command="npx",
    args=[
        "-y",
        "@openbnb/mcp-server-airbnb",
        "--ignore-robots-txt"
    ]
)

class AirbnbMCPClient(BaseMCPClient):
    """Specific client for Airbnb MCP server."""
    
    def __init__(self):
        super().__init__(AIRBNB_SERVER)

    @classmethod
    def search_sync(cls, location: str, **kwargs) -> str:
        """
        Sync wrapper for LangChain tools.
        Creates its own client instance, runs the async search, returns JSON.
        """
        client = cls()
        result = asyncio.run(client.search(location, **kwargs))
        return json.dumps(result, indent=2)

    async def search(
        self,
        location: str,
        check_in: str  = None,
        check_out: str = None,
        adults: int    = 2,
        children: int  = 0,
        infants: int   = 0,
        pets: int      = 0,
    ) -> dict[str, Any]:
        """Specific search implementation for Airbnb."""
        tool_args = {
            "location":  location,
            "adults":    adults,
            "children":  children,
            "infants":   infants,
            "pets":      pets,
        }
        if check_in:
            tool_args["check_in"]  = check_in
        if check_out:
            tool_args["check_out"] = check_out

        return await self.call_tool("airbnb_search", tool_args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "═" * 60)
    print("SEARCHING AIRBNB (VIA CLASSMETHOD)")
    print("═" * 60)
    results = AirbnbMCPClient.search_sync(location="Miami, FL", check_in="2026-06-01", check_out="2026-06-07")
    print(results)
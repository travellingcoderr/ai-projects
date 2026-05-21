"""
mcp_aviationstack.py
---------------------
Aviationstack MCP Client implementation.
"""

import asyncio
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from mcp import StdioServerParameters
from .base import BaseMCPClient

load_dotenv()
logger = logging.getLogger(__name__)

# ── API Key ──────────────────────────────────────────────────────────────
AVIATION_API_KEY = os.getenv("AVIATION_STACK_API_KEY")

class AviationstackMCPClient(BaseMCPClient):
    """Specific client for Aviationstack MCP server."""
    
    def __init__(self):
        if not AVIATION_API_KEY:
            raise EnvironmentError("AVIATION_STACK_API_KEY not set.")
            
        params = StdioServerParameters(
            command="uvx",
            args=["aviationstack-mcp"],
            env={
                "AVIATION_STACK_API_KEY": AVIATION_API_KEY,
                "PATH": os.environ.get("PATH", "")
            }
        )
        super().__init__(params)

    @classmethod
    def search_sync(cls, departure_iata: str, **kwargs) -> str:
        """
        Sync wrapper for LangChain tools.
        Creates its own client instance, runs the async search, returns JSON.
        """
        client = cls()
        result = asyncio.run(client.search(departure_iata, **kwargs))
        return json.dumps(result, indent=2)

    async def search(
        self,
        departure_iata: str,
        arrival_iata: str   = None,
        flight_date: str    = None,
        airline_name: str   = None,
        flight_status: str  = None,
        limit: int          = 5,
    ) -> dict[str, Any]:
        """Specific search implementation for flights."""
        tool_args = {
            "dep_iata": departure_iata,
            "limit":    limit,
        }
        if arrival_iata:
            tool_args["arr_iata"]       = arrival_iata
        if flight_date:
            tool_args["flight_date"]    = flight_date
        if airline_name:
            tool_args["airline_name"]   = airline_name
        if flight_status:
            tool_args["flight_status"]  = flight_status

        # Logic for tool selection
        tools = await self.list_tools()
        tool_names = [t["name"] for t in tools]
        
        preferred_tools = [
            "flight_arrival_departure_schedule",
            "get_flights",
            "search_flights"
        ]
        
        flight_tool = next(
            (name for name in preferred_tools if name in tool_names),
            next((name for name in tool_names if "flight" in name.lower()), tool_names[0])
        )

        return await self.call_tool(flight_tool, tool_args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        print("\n" + "═" * 60)
        print("SEARCHING FLIGHTS (VIA CLASSMETHOD)")
        print("═" * 60)
        results = AviationstackMCPClient.search_sync(departure_iata="JFK", arrival_iata="MIA", flight_date="2026-05-01")
        print(results)
    except Exception as e:
        print(f"Error: {e}")
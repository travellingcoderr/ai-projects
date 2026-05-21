"""
base.py
-------
Core abstractions for MCP Clients.

This file defines the contract (IMCPClient) and the base implementation 
(BaseMCPClient) to handle common MCP stdio and session logic.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

class IMCPClient(ABC):
    """
    The Interface (Contract). 
    Defines what any MCP client must be able to do.
    """
    
    @abstractmethod
    async def list_tools(self) -> List[dict]:
        """Discover available tools on the MCP server."""
        pass

    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a specific tool on the MCP server."""
        pass


class BaseMCPClient(IMCPClient, ABC):
    """
    The Base Abstract Class.
    Implements the shared plumbing for stdio communication and session management.
    """
    
    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params

    async def _get_session(self):
        """Context manager for the stdio connection and session."""
        # Note: We use a context manager pattern inside call_tool/list_tools 
        # to ensure the connection is cleaned up after every call.
        pass

    async def list_tools(self) -> List[dict]:
        """Shared implementation for tool discovery."""
        logger.info(f"Connecting to MCP server for tool discovery: {self.server_params.command}")
        async with stdio_client(self.server_params) as (read, write):
            logger.info("Stdio connection established. Initializing session...")
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("Session initialized. Listing tools...")
                tools_result = await session.list_tools()
                
                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputs": tool.inputSchema
                    }
                    for tool in tools_result.tools
                ]

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Shared implementation for executing a tool."""
        logger.info(f"Connecting to MCP server to call tool: {tool_name}")
        
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Check tool existence or just call directly
                result = await session.call_tool(
                    name=tool_name,
                    arguments=arguments
                )
                
                # Extract text content and parse as JSON if possible
                raw = result.content[0].text if result.content else "{}"
                
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {"raw_response": raw}

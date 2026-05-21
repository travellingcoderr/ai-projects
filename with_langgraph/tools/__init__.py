"""
tools/__init__.py
-----------------
Exports a single `tools` list consumed in two places:
  1. llm.bind_tools(tools)   — tells the LLM what functions are available
  2. ToolNode(tools)         — executes the tools when the LLM calls them

This is identical in purpose to with_langchain/tools/__init__.py, but
we use plain @tool-decorated functions instead of StructuredTool wrappers.
"""

from .weather import get_weather
from .calendar import get_calendar_events

tools = [get_weather, get_calendar_events]

__all__ = ["tools"]

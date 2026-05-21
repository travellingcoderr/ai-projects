"""
weather.py — Weather tool for LangGraph agent
-----------------------------------------------
LangGraph uses the same `@tool` decorator from langchain_core as LangChain.

The difference vs with_langchain/tools/weather.py:
  with_langchain : StructuredTool(func=..., args_schema=WeatherInput)
  with_langgraph : just @tool — cleaner, and ToolNode understands it natively

In production you'd swap the mock return for a real API call, e.g.:
  import requests
  resp = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=...")
  return resp.json()["weather"][0]["description"]
"""

import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: The name of the city to get weather for.
    """
    if not city or not city.strip():
        return "[INPUT ERROR] City name cannot be empty."
    logger.info(f"[Weather Tool] Fetching weather for: {city}")
    # Replace with a real weather API call in production
    return f"It's 72°F and sunny in {city}."

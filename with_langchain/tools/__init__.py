from .weather import weather_tool
from .calendar import calendar_tool

# Collect all tools into a single list for the agent
tools = [weather_tool, calendar_tool]

__all__ = ["tools", "weather_tool", "calendar_tool"]

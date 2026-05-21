"""
calendar.py — Calendar tool for LangGraph agent
-------------------------------------------------
Same calendar tool from prior agents, now decorated with @tool
for native LangGraph / ToolNode compatibility.

In production you'd call a real calendar API, e.g. Google Calendar:
  from googleapiclient.discovery import build
  service = build("calendar", "v3", credentials=creds)
  events = service.events().list(calendarId="primary", timeMin=...).execute()
"""

import logging
from datetime import datetime
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_calendar_events(date: str) -> str:
    """Get calendar events for a specific date.

    Args:
        date: The date to get events for, in YYYY-MM-DD format.
    """
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"[INPUT ERROR] Invalid date '{date}'. Please use YYYY-MM-DD format."

    logger.info(f"[Calendar Tool] Fetching events for: {date}")
    # Replace with a real Google Calendar / Outlook API call in production
    return f"On {date} you have: 9am Standup, 2pm Product Review."

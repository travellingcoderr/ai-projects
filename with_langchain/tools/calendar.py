import logging
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from ..utils import with_timeout, with_error_recovery

logger = logging.getLogger(__name__)

class CalendarInput(BaseModel):
    date: str = Field(description="The date to get events for in YYYY-MM-DD format")

@with_error_recovery
@with_timeout(seconds=5)
def get_calendar_events(date: str) -> str:
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date '{date}'. Use YYYY-MM-DD format.")
    logger.info(f"Fetching calendar for: {date}")
    return f"On {date} you have: 9am Standup, 2pm Product Review"

calendar_tool = StructuredTool(
    name="get_calendar_events",
    description="Get calendar events for a specific date",
    func=get_calendar_events,
    args_schema=CalendarInput
)

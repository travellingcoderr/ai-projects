import logging
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from ..utils import with_timeout, with_error_recovery

logger = logging.getLogger(__name__)

class WeatherInput(BaseModel):
    city: str = Field(description="The name of the city to get weather for")

@with_error_recovery
@with_timeout(seconds=5)
def get_weather(city: str) -> str:
    if not city or not city.strip():
        raise ValueError("City name cannot be empty")
    logger.info(f"Fetching weather for: {city}")
    return f"It's 72°F and sunny in {city}"

weather_tool = StructuredTool(
    name="get_weather",
    description="Get the current weather for a given city",
    func=get_weather,
    args_schema=WeatherInput
)

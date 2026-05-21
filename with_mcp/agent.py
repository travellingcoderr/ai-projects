"""
agent.py — LangChain Agent with MCP Tools
-------------------------------------------
Wraps the Airbnb and Aviationstack MCP connectors as LangChain
StructuredTools, then runs a LangChain agent that can call them.

Data flow:
  User → LangChain Agent → StructuredTool → MCP Connector → MCP Server → Raw Results

This is the same LangChain pattern as our weather/calendar agent —
the only difference is the tool functions now call MCP servers
instead of returning mock data.
"""

import logging
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# ── Import our MCP connectors ─────────────────────────────────────────────
from .mcps import AirbnbMCPClient, AviationstackMCPClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS — define the inputs each LangChain tool accepts
# LangChain uses these to auto-generate JSON schema for the LLM
# ════════════════════════════════════════════════════════════════════════

class AirbnbSearchInput(BaseModel):
    location: str = Field(
        description="City or area to search for Airbnb listings. e.g. 'Miami, FL'"
    )
    check_in: str = Field(
        default=None,
        description="Check-in date in YYYY-MM-DD format. e.g. '2026-06-01'"
    )
    check_out: str = Field(
        default=None,
        description="Check-out date in YYYY-MM-DD format. e.g. '2026-06-07'"
    )
    adults: int = Field(
        default=1,
        description="Number of adult guests"
    )


class FlightSearchInput(BaseModel):
    departure_iata: str = Field(
        description="IATA airport code for departure. e.g. 'JFK', 'LAX', 'TPA'"
    )
    arrival_iata: str = Field(
        default=None,
        description="IATA airport code for arrival. e.g. 'MIA', 'ORD'"
    )
    flight_date: str = Field(
        default=None,
        description="Date of the flight in YYYY-MM-DD format"
    )
    airline_name: str = Field(
        default=None,
        description="Filter by airline name. e.g. 'Delta', 'American Airlines'"
    )
    flight_status: str = Field(
        default=None,
        description="Filter by status: scheduled | active | landed | cancelled"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of results to return"
    )


# ════════════════════════════════════════════════════════════════════════
# STRUCTURED TOOLS — wrap MCP connectors as LangChain tools
#
# Same pattern as our weather/calendar agent:
#   StructuredTool(func=..., args_schema=...)
# Except func now calls an MCP server instead of mock data
# ════════════════════════════════════════════════════════════════════════

airbnb_tool = StructuredTool(
    name="search_airbnb",
    description="Search for Airbnb listings in a specific location with dates and guest count.",
    func=AirbnbMCPClient.search_sync,
    args_schema=AirbnbSearchInput
)

flights_tool = StructuredTool(
    name="search_flights",
    description="Search for real-time flight information between airports.",
    func=AviationstackMCPClient.search_sync,
    args_schema=FlightSearchInput
)

tools = [airbnb_tool, flights_tool]


# ════════════════════════════════════════════════════════════════════════
# LLM — streaming enabled so tokens print live
# ════════════════════════════════════════════════════════════════════════

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)


# ════════════════════════════════════════════════════════════════════════
# MEMORY — remembers conversation across turns
# ════════════════════════════════════════════════════════════════════════

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="input",
    output_key="output"
)


# ════════════════════════════════════════════════════════════════════════
# PROMPT
# ════════════════════════════════════════════════════════════════════════

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a travel assistant with access to Airbnb listings and "
        "flight data. Today's date is {today}. "
        "When calling flight tools, always convert city names to IATA codes first. "
        "Return raw results clearly — do not summarize, just present the data."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


# ════════════════════════════════════════════════════════════════════════
# AGENT + EXECUTOR
# ════════════════════════════════════════════════════════════════════════

agent = create_tool_calling_agent(llm, tools, prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=5,
    max_execution_time=60,              # MCP servers can be slow to spin up
    handle_parsing_errors=(
        "I had trouble reading that response. Let me try again."
    ),
    return_intermediate_steps=True
)


# ════════════════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════════════════

def chat(user_input: str):
    print(f"\n{'═' * 60}")
    print(f"You: {user_input}")
    print(f"{'═' * 60}\n")

    result = executor.invoke({
        "input": user_input,
        "today": datetime.now().strftime("%Y-%m-%d")
    })

    # Print tool calls made during this turn
    if result.get("intermediate_steps"):
        print(f"\n{'─' * 60}")
        print("Tools Called:")
        for action, observation in result["intermediate_steps"]:
            print(f"\n  Tool   : {action.tool}")
            print(f"  Input  : {action.tool_input}")
            print(f"  Result : {observation[:300]}...")   # trim long results
        print(f"{'─' * 60}")

    return result["output"]


# ════════════════════════════════════════════════════════════════════════
# DEMO — tests each MCP separately then combines them
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Test 1 — Airbnb MCP only
    chat("Search for Airbnb listings in Miami for 2 adults, June 1-7 2026")

    # Test 2 — Aviationstack MCP only
    chat("Show me flights from JFK to Miami on June 1st 2026")

    # Test 3 — memory: agent knows Miami + June 1 from prior turns
    chat("What airlines fly that route?")
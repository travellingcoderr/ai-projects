"""
supervisor.py — Multi-Agent Supervisor Pattern (LangGraph)
-----------------------------------------------------------
This is the ADVANCED LangGraph pattern cited directly in the PwC job spec:
  "Building and integrating multi-agent frameworks (e.g., LangGraph, LangFlow)
   to enable autonomous AI task execution."

Architecture — Supervisor orchestrates two specialist agents:

  START
    │
    ▼
  [supervisor] ──── FINISH ──────────────────────────────► END
       │
  ┌────┴──────────────────────────┐
  │                               │
  ▼                               ▼
[researcher]                  [analyst]
  (fetches data                (interprets data
   using tools)                 no tools needed)
  │                               │
  └──── back to [supervisor] ◄────╯

Why this pattern matters:
  - Each agent has a focused role → better outputs than one big agent
  - Supervisor uses structured output → no hallucinated routing
  - create_react_agent() builds sub-graphs automatically → composable
  - Any specialist can be swapped out without touching the others
  - Scales to N specialists with the same supervisor code

How this maps to PwC work:
  Supervisor  → orchestrates the pipeline
  Researcher  → calls APIs / databases (RAG retrieval, tool use)
  Analyst     → interprets results (LLM reasoning over gathered context)
  You'd add:  Formatter, Reviewer, QA agents for full production pipelines

Key new concepts vs graph.py:
  - Structured output (with_structured_output) → typed routing decisions
  - create_react_agent()  → shorthand for building a sub-agent graph
  - SupervisorState       → extended state with a routing field
  - Conditional edges     → reads state["next"] set by supervisor
"""

import logging
from datetime import datetime
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from .state import SupervisorState
from .tools import tools

load_dotenv()
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT SCHEMA — supervisor's routing decision
#
# Using structured output prevents the LLM from returning free text
# that we'd have to parse. The LLM MUST return a valid SupervisorDecision.
# This is a production best practice for any routing agent.
# ════════════════════════════════════════════════════════════════════════

class SupervisorDecision(BaseModel):
    """The supervisor's decision about which agent to call next."""
    next: Literal["researcher", "analyst", "FINISH"] = Field(
        description="Which agent to route to next, or FINISH when the task is complete."
    )
    reasoning: str = Field(
        description="One sentence explaining why this agent was chosen."
    )


# ════════════════════════════════════════════════════════════════════════
# SPECIALIST AGENTS
#
# create_react_agent() is a LangGraph shorthand that builds a mini
# StateGraph identical to what we built manually in graph.py:
#   agent_node → (conditional) → ToolNode → agent_node → ...
#
# The state_modifier injects a system-level instruction for each specialist.
# This is how you give each agent a focused role without separate prompts.
# ════════════════════════════════════════════════════════════════════════

# Researcher: has tools, fetches raw data
researcher_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=tools,
    state_modifier=(
        "You are a research specialist. Use your tools to fetch raw, factual data — "
        "weather conditions and calendar events. Return the raw facts exactly as retrieved. "
        "Do not interpret, summarize, or make recommendations. Just fetch and report."
    )
)

# Analyst: no tools, reasons over gathered data
analyst_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=[],   # Analyst reasons — no API calls needed
    state_modifier=(
        "You are an analysis specialist. Based on the research data already present "
        "in the conversation, provide clear, actionable recommendations and insights. "
        "Do NOT call any tools — synthesize the information you already have. "
        "Be concise and practical."
    )
)


# ════════════════════════════════════════════════════════════════════════
# SUPERVISOR NODE — the orchestrator
#
# The supervisor reads the entire conversation, then uses structured output
# to decide which specialist runs next. It does NOT add messages to state —
# it only sets state["next"] to guide the routing function.
# ════════════════════════════════════════════════════════════════════════

SUPERVISOR_SYSTEM = """
You are a supervisor coordinating two specialist agents to answer the user's question.

Available agents:
  researcher — fetches real-time data (weather, calendar) using tools.
               Use when the user needs current factual information.
  analyst    — interprets gathered data and gives recommendations (no tools).
               Use AFTER the researcher has fetched relevant data.

Decision rules:
  1. If no data has been fetched yet and the user needs it → route to 'researcher'
  2. If raw data is available and interpretation is needed → route to 'analyst'
  3. If the analyst has given a complete, actionable answer → return 'FINISH'
  4. Do not route to the same agent twice in a row without new information.
  5. Return 'FINISH' if the task cannot be completed with available tools.
"""

supervisor_llm = ChatOpenAI(model="gpt-4o", temperature=0)
supervisor_with_structured_output = supervisor_llm.with_structured_output(SupervisorDecision)


def supervisor_node(state: SupervisorState) -> dict:
    """
    Reads the full conversation and decides which specialist runs next.

    Returns a dict updating only `next` — no new message is added here.
    The routing function will read state["next"] to pick the next node.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM + f"\n\nToday is {today}."),
        *state["messages"]
    ]
    decision: SupervisorDecision = supervisor_with_structured_output.invoke(messages)
    logger.info(f"[supervisor] → {decision.next} | {decision.reasoning}")
    return {"next": decision.next}


# ════════════════════════════════════════════════════════════════════════
# SPECIALIST WRAPPER NODES
#
# These thin wrappers invoke the sub-agents (mini graphs) and return
# their output messages to the shared SupervisorState.
# The sub-agent's messages get appended by the add_messages reducer.
# ════════════════════════════════════════════════════════════════════════

def researcher_node(state: SupervisorState) -> dict:
    """Invokes the researcher sub-agent and returns its messages."""
    logger.info("[researcher] Starting data retrieval...")
    result = researcher_agent.invoke(state)
    return {"messages": result["messages"]}


def analyst_node(state: SupervisorState) -> dict:
    """Invokes the analyst sub-agent and returns its messages."""
    logger.info("[analyst] Starting analysis...")
    result = analyst_agent.invoke(state)
    return {"messages": result["messages"]}


# ════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTION — reads state["next"] set by supervisor_node
# ════════════════════════════════════════════════════════════════════════

def route(state: SupervisorState) -> str:
    """Returns the next node name, or END if the supervisor said FINISH."""
    decision = state.get("next")
    if decision == "FINISH":
        return END
    return decision   # "researcher" or "analyst"


# ════════════════════════════════════════════════════════════════════════
# BUILD SUPERVISOR GRAPH
# ════════════════════════════════════════════════════════════════════════

def build_supervisor_graph():
    """
    Assembles the supervisor multi-agent graph.

    All specialist nodes report back to supervisor when done.
    Supervisor decides whether to call another specialist or FINISH.
    """
    sg = StateGraph(SupervisorState)

    sg.add_node("supervisor",  supervisor_node)
    sg.add_node("researcher",  researcher_node)
    sg.add_node("analyst",     analyst_node)

    # Always start with the supervisor
    sg.set_entry_point("supervisor")

    # Supervisor decides what runs next
    sg.add_conditional_edges(
        "supervisor",
        route,
        {
            "researcher": "researcher",
            "analyst":    "analyst",
            END:          END
        }
    )

    # Both specialists always report back to supervisor when done
    sg.add_edge("researcher", "supervisor")
    sg.add_edge("analyst",    "supervisor")

    return sg.compile(checkpointer=MemorySaver())


supervisor_graph = build_supervisor_graph()
SUPERVISOR_THREAD = {"configurable": {"thread_id": "supervisor-demo-1"}, "recursion_limit": 15}


# ════════════════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════════════════

def run_supervisor(user_input: str) -> str:
    """
    Runs the supervisor graph and streams events.

    Watch the logs — you'll see the supervisor routing decisions printed
    at each step, which is invaluable for understanding the flow.
    """
    print(f"\n{'═' * 60}")
    print(f"You: {user_input}")
    print(f"{'═' * 60}\n")

    final_answer = ""

    for event in supervisor_graph.stream(
        {"messages": [HumanMessage(content=user_input)], "next": None},
        config=SUPERVISOR_THREAD,
        stream_mode="values"
    ):
        last = event["messages"][-1]
        if (
            last.__class__.__name__ == "AIMessage"
            and last.content
            and not getattr(last, "tool_calls", [])
        ):
            final_answer = last.content

    print(f"\nFinal Answer:\n{final_answer}\n")
    return final_answer


# ════════════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # The supervisor should:
    # 1. Route to researcher (needs weather + calendar data)
    # 2. Route to analyst (interpret: should the meeting be outside?)
    # 3. Return FINISH
    run_supervisor(
        "What's the weather in Tampa today and do I have any meetings? "
        "Based on the data, should I hold my 9am standup outside?"
    )

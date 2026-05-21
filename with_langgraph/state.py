"""
state.py — LangGraph Agent State
----------------------------------
In LangGraph, every node reads from and writes to a shared `state` dict.
The state definition using TypedDict + Annotated reducers is the MOST
IMPORTANT concept in LangGraph — it replaces LangChain's
ConversationBufferMemory and the ad-hoc message list from the raw approach.

Key insight: `add_messages` is a REDUCER — instead of replacing the
messages list each time a node returns, it APPENDS new messages. This is
what makes the agent loop work without any extra bookkeeping.

Compare across all three approaches:
  without_langchain : manually managed `chat_history = []`
  with_langchain    : ConversationBufferMemory(memory_key="chat_history")
  with_langgraph    : Annotated[list, add_messages] — the graph handles it
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The shared state that flows through every node of the graph.

    Fields
    ------
    messages : full conversation history — append-only via add_messages reducer

    How reducers work
    -----------------
    When a node returns {"messages": [new_msg]}, LangGraph does NOT replace
    state["messages"]. Instead it CALLS add_messages(state["messages"], [new_msg])
    which appends. This means every node just returns what it adds — never
    the full list.

    You can add more fields here as your agent grows, e.g.:
      tool_outputs : list[str]   — store raw tool results separately
      current_task : str         — track what the agent is working on
      error_count  : int         — track failures for retry logic
    """
    messages: Annotated[list, add_messages]


class SupervisorState(AgentState):
    """
    Extended state for the multi-agent supervisor pattern.

    Adds a `next` field that the supervisor node sets to decide
    which specialist agent runs next: "researcher", "analyst", or "FINISH".

    This is the canonical LangGraph way to pass routing decisions between
    nodes without polluting the messages list with control-flow data.
    """
    next: Optional[str]

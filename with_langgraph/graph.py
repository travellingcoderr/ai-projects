"""
graph.py — LangGraph StateGraph construction
---------------------------------------------
This is the CORE file. It replaces AgentExecutor from with_langchain with
an explicit, inspectable graph where YOU define:

  Nodes  : what computation happens
  Edges  : what node runs next (fixed transitions)
  Cond.  : conditional routing — LLM output decides the next step

The ReAct loop that LangChain hides inside AgentExecutor is here fully
visible as a simple two-node cycle:

  START
    │
    ▼
  [agent_node]  ◄────────────────────────────────╮
    │                                             │
    ▼ (conditional edge via should_continue)      │
  has_tool_calls? ──YES──► [tool_node] ───────────╯
    │
   NO
    ▼
   END

Key LangGraph primitives used here:

  StateGraph   — the graph builder (like a blueprint)
  add_node     — register a callable as a graph node
  add_edge     — fixed unconditional transition
  add_conditional_edges — routing based on a function of state
  set_entry_point — which node runs first
  compile      — locks the graph; returns a runnable Pregel object
  MemorySaver  — in-memory checkpointer for thread-based persistence

Compare:
  with_langchain : AgentExecutor(agent, tools, memory, max_iterations=5)
    → you configure boxes; the loop is hidden
  with_langgraph : explicit StateGraph with nodes + edges
    → you OWN the topology; full control and visibility
"""

import logging
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .tools import tools

load_dotenv()
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# LLM — bind tools so the model knows what functions it can call
#
# bind_tools() attaches the tool JSON schemas to every LLM call.
# The LLM then returns AIMessage.tool_calls when it wants to use them.
# This replaces the manual TOOLS = [...] list from without_langchain.
# ════════════════════════════════════════════════════════════════════════

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ════════════════════════════════════════════════════════════════════════
# NODE 1: AGENT — sends messages to the LLM; may produce tool_calls
#
# Every node is just a plain Python function:
#   Input  : the full AgentState
#   Output : a PARTIAL state update (dict) — only what changed
#
# The `add_messages` reducer on state["messages"] will APPEND the new
# AIMessage returned here — it never replaces the full history.
# ════════════════════════════════════════════════════════════════════════

def agent_node(state: AgentState) -> dict:
    """
    The 'brain' node: invokes the LLM with the current message history.

    LangChain equivalent: the inner agent callable inside AgentExecutor.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    system = SystemMessage(
        content=(
            f"You are a helpful assistant with access to weather and calendar tools. "
            f"Today's date is {today}. If a tool returns an error, explain it to the "
            f"user and suggest alternatives."
        )
    )

    # System message injected at the front; rest is existing history
    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)

    tool_call_count = len(response.tool_calls) if hasattr(response, "tool_calls") else 0
    logger.info(f"[agent_node] LLM responded | tool_calls={tool_call_count}")

    # Return only the new message — add_messages handles the append
    return {"messages": [response]}


# ════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE FUNCTION: should_continue
#
# After agent_node runs, LangGraph calls this function to decide what
# happens next. It inspects the last message and routes accordingly.
#
# This logic is HIDDEN inside AgentExecutor in LangChain. In LangGraph
# you write it explicitly — giving you full visibility and the ability
# to add custom conditions (e.g., "route to human review if confidence < 0.8").
# ════════════════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> str:
    """
    Inspects the last LLM message and decides what node runs next.

    Returns "tools" → ToolNode executes the requested tool calls
    Returns END     → no more tool calls, the graph finishes
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"[router] → tools: {tool_names}")
        return "tools"
    logger.info("[router] → END (no tool calls)")
    return END


# ════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Assembles and compiles the StateGraph.

    ToolNode is a LangGraph prebuilt that:
      1. Reads tool_calls from the last AIMessage
      2. Executes the matching tool functions (from our `tools` list)
      3. Wraps each result in a ToolMessage and appends it to state

    This replaces:
      - The manual FUNCTION_MAP + for loop in without_langchain/agent.py
      - The internal tool runner inside AgentExecutor in with_langchain/agent.py

    MemorySaver (checkpointer):
      - Stores a snapshot of state after every node execution
      - Each thread_id gets its own isolated conversation history
      - Swap for SqliteSaver / RedisSaver in production for real persistence
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────────────────
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))   # built-in tool executor

    # ── Entry point ────────────────────────────────────────────────────
    graph.set_entry_point("agent")

    # ── Conditional edge: agent → (tools | END) ────────────────────────
    # After agent_node runs, call should_continue() to decide next step
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",   # LLM wants to call tools → go to tool_node
            END: END            # LLM has a final answer → stop
        }
    )

    # ── Fixed edge: tools → agent (the loop) ──────────────────────────
    # After ToolNode runs, always send results back to the agent
    graph.add_edge("tools", "agent")

    # ── Compile with in-memory checkpointer ───────────────────────────
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Pre-compiled graph — import this in agent.py and supervisor.py
graph = build_graph()

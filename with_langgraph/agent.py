"""
WITH LANGGRAPH — Explicit graph-based ReAct agent
---------------------------------------------------
LangGraph handles:
  - Explicit state machine (StateGraph with typed AgentState)
  - Tool routing via ToolNode (no manual FUNCTION_MAP needed)
  - The agent loop as a graph cycle (agent → tools → agent → ...)
  - Streaming via graph.stream() — no callback handlers needed
  - Thread-based memory via MemorySaver checkpointer
  - Iteration limits via recursion_limit in config

Side-by-side comparison of all three approaches for the SAME agent:

  ┌─────────────────────┬──────────────────────────────┬────────────────────────────┐
  │ Concern             │ without_langchain            │ with_langchain             │ with_langgraph             │
  ├─────────────────────┼──────────────────────────────┼────────────────────────────┤
  │ Tool schema         │ Manually written JSON        │ Pydantic → auto-generated  │ @tool → auto-generated     │
  │ Tool routing        │ FUNCTION_MAP dict + if/else  │ AgentExecutor internal     │ ToolNode (prebuilt)        │
  │ Agent loop          │ while True:                  │ AgentExecutor internal     │ graph edge: tools→agent    │
  │ Memory              │ chat_history = []            │ ConversationBufferMemory   │ MemorySaver checkpointer   │
  │ Streaming           │ client.chat...stream=True    │ StreamingStdOutCallback    │ graph.stream()             │
  │ Max iterations      │ if iterations >= MAX:        │ max_iterations=5           │ recursion_limit=10         │
  │ Topology visibility │ Hidden in while loop         │ Hidden in AgentExecutor    │ Fully explicit graph       │
  └─────────────────────┴──────────────────────────────┴────────────────────────────┘

Why LangGraph for PwC?
  - Multi-agent orchestration needs explicit topology (supervisor patterns)
  - Human-in-the-loop checkpoints require inspectable state
  - Production AI systems need controllable, observable loops
  - LangGraph is the foundation LangFlow, LangSmith, and LangServe are built on
"""

import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from .graph import graph

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# THREAD CONFIG — replaces passing `memory=` to AgentExecutor
#
# Each unique thread_id is a separate, isolated conversation history.
# The MemorySaver checkpointer stores state per-thread automatically.
#
# Why this is better than ConversationBufferMemory:
#   - Multiple conversations run in parallel with no state collision
#   - You can fork a conversation from any checkpoint
#   - Swap MemorySaver → SqliteSaver for real database persistence
#   - You can inspect / replay the full state at any point in time
# ════════════════════════════════════════════════════════════════════════

THREAD_CONFIG = {
    "configurable": {
        "thread_id": "demo-thread-1"
    },
    "recursion_limit": 10    # replaces max_iterations=5 from AgentExecutor
}


# ════════════════════════════════════════════════════════════════════════
# STREAMING RUNNER
#
# graph.stream() yields one state snapshot after EACH node execution.
# stream_mode="values" gives the FULL state after each step (vs "updates"
# which gives only the delta — useful for debugging).
#
# This replaces executor.invoke() + StreamingStdOutCallbackHandler from
# with_langchain. You now control exactly what gets printed and when.
# ════════════════════════════════════════════════════════════════════════

def run_agent(user_input: str, config: dict = THREAD_CONFIG) -> str:
    """
    Runs the LangGraph agent and streams the response.

    Each event from graph.stream() is a full AgentState snapshot.
    We look at the last message in each snapshot — when it's an AIMessage
    with content (not a tool call), that's the agent's response.
    """
    print("\nAssistant: ", end="", flush=True)
    final_answer = ""

    for event in graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="values"    # full state after each node
    ):
        last_message = event["messages"][-1]

        # AIMessage with content = the agent is speaking to the user
        # AIMessage with tool_calls only = intermediate step (skip printing)
        # ToolMessage = tool result (skip printing)
        if (
            last_message.__class__.__name__ == "AIMessage"
            and last_message.content
            and not getattr(last_message, "tool_calls", [])
        ):
            # Overwrite previous partial output (streaming effect)
            print(f"\rAssistant: {last_message.content}", end="", flush=True)
            final_answer = last_message.content

    print()  # newline after streaming completes
    return final_answer


# ════════════════════════════════════════════════════════════════════════
# INSPECT STATE — unique to LangGraph (no equivalent in LangChain)
#
# At any point you can call graph.get_state(config) to inspect:
#   - Full message history
#   - What node would run next
#   - All checkpoint metadata
#
# This enables human-in-the-loop patterns, auditing, and debugging.
# ════════════════════════════════════════════════════════════════════════

def inspect_state(config: dict = THREAD_CONFIG):
    """Prints the current graph state — unique LangGraph capability."""
    state = graph.get_state(config)
    print(f"\n{'─' * 60}")
    print("Graph State Snapshot:")
    print(f"  Messages in history : {len(state.values.get('messages', []))}")
    print(f"  Next node to run    : {state.next or 'None (graph completed)'}")
    print(f"{'─' * 60}")
    return state


# ════════════════════════════════════════════════════════════════════════
# MULTI-TURN DEMO — same three turns used in without_langchain + with_langchain
# ════════════════════════════════════════════════════════════════════════

def chat(user_input: str):
    print(f"\n{'═' * 60}")
    print(f"You: {user_input}")
    print(f"{'═' * 60}")
    return run_agent(user_input)


if __name__ == "__main__":
    # Turn 1 — LLM calls both weather and calendar tools
    chat("What's the weather in Tampa and what meetings do I have today?")

    # Turn 2 — MemorySaver remembers Turn 1 via thread_id (no memory object needed)
    chat("Is it a good day to hold the 9am meeting outside?")

    # Turn 3 — tests error handling in calendar tool (bad date format)
    chat("What's on my calendar for the 25th of April?")

    # Inspect the full graph state after all turns
    inspect_state()

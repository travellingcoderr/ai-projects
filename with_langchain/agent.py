"""
WITH LANGCHAIN — Production-grade agent
----------------------------------------
LangChain handles:
  - Tool schema generation (StructuredTool + Pydantic)
  - Tool routing (AgentExecutor)
  - The agent loop (AgentExecutor)
  - Streaming (streaming=True + StreamingStdOutCallbackHandler)
  - Timeouts per-tool (@with_timeout) + whole-agent (max_execution_time)
  - Max iterations (max_iterations)
  - Error recovery (@with_error_recovery + handle_parsing_errors)
  - Memory (ConversationBufferMemory)
"""

from datetime import datetime
from dotenv import load_dotenv

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Local imports
from .tools import tools

load_dotenv()

# ════════════════════════════════════════════════════════════════════════
# LLM SETUP
# ════════════════════════════════════════════════════════════════════════

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# ════════════════════════════════════════════════════════════════════════
# MEMORY SETUP
# ════════════════════════════════════════════════════════════════════════

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="input"
)

# ════════════════════════════════════════════════════════════════════════
# PROMPT SETUP
# ════════════════════════════════════════════════════════════════════════

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant with access to weather and calendar tools. "
        "Today's date is {today}. If a tool returns an error, explain it to the "
        "user and suggest alternatives."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# ════════════════════════════════════════════════════════════════════════
# AGENT + EXECUTOR SETUP
# ════════════════════════════════════════════════════════════════════════

agent = create_tool_calling_agent(llm, tools, prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=5,
    max_execution_time=30,
    handle_parsing_errors=(
        "I had trouble understanding that. Let me try again."
    ),
    return_intermediate_steps=True
)

# ════════════════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════════════════

def chat(user_input: str):
    print(f"\n{'─' * 60}")
    print(f"You: {user_input}")
    print(f"{'─' * 60}")

    result = executor.invoke({
        "input": user_input,
        "today": datetime.now().strftime("%Y-%m-%d")
    })

    if result.get("intermediate_steps"):
        print("\n── Tool Calls Made ──────────────────────────────────")
        for action, observation in result["intermediate_steps"]:
            print(f"  Tool   : {action.tool}")
            print(f"  Input  : {action.tool_input}")
            print(f"  Result : {observation}")
        print("─────────────────────────────────────────────────────")

    return result["output"]

# ════════════════════════════════════════════════════════════════════════
# MULTI-TURN DEMO
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    chat("What's the weather in Tampa and what meetings do I have today?")
    chat("Is it a good day to hold the 9am meeting outside?")
    chat("What's on my calendar for the 25th of April?")
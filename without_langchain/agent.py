"""
WITHOUT LANGCHAIN — Raw OpenAI agent
-------------------------------------
Manually handles:
  - Tool definition (JSON schema)
  - Tool routing (function_map)
  - The agent loop (while True)
  - Message history
  - Streaming
  - Timeouts (per-tool via signal)
  - Max iterations
  - Error recovery
"""

import json
import signal
import logging
import functools
from datetime import datetime

import openai
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

client = openai.OpenAI()

# ════════════════════════════════════════════════════════════════════════
# TIMEOUT — decorator that enforces max execution time per tool
# ════════════════════════════════════════════════════════════════════════

class ToolTimeoutError(Exception):
    pass

def with_timeout(seconds: int):
    """Kills a tool function if it runs longer than `seconds`."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def _handler(signum, frame):
                raise ToolTimeoutError(
                    f"Tool '{func.__name__}' timed out after {seconds}s"
                )
            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
        return wrapper
    return decorator

# ════════════════════════════════════════════════════════════════════════
# ERROR RECOVERY — decorator that catches exceptions and returns them
# as strings so the LLM can read and adapt, rather than crashing
# ════════════════════════════════════════════════════════════════════════

def with_error_recovery(func):
    """Returns errors as readable strings instead of raising them."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ToolTimeoutError as e:
            msg = f"[TIMEOUT ERROR] {e}. Try a different input."
            logger.error(msg)
            return msg
        except ConnectionError as e:
            msg = f"[CONNECTION ERROR] Could not reach service: {e}"
            logger.error(msg)
            return msg
        except ValueError as e:
            msg = f"[INPUT ERROR] Bad input: {e}"
            logger.error(msg)
            return msg
        except Exception as e:
            msg = f"[UNEXPECTED ERROR] {type(e).__name__}: {e}"
            logger.error(msg)
            return msg
    return wrapper

# ════════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS
# ════════════════════════════════════════════════════════════════════════

@with_error_recovery
@with_timeout(seconds=5)
def get_weather(city: str) -> str:
    if not city or not city.strip():
        raise ValueError("City name cannot be empty")
    logger.info(f"Fetching weather for: {city}")
    # Replace with a real API call e.g. requests.get(weather_api_url)
    return f"It's 72°F and sunny in {city}"

@with_error_recovery
@with_timeout(seconds=5)
def get_calendar_events(date: str) -> str:
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date '{date}'. Use YYYY-MM-DD format.")
    logger.info(f"Fetching calendar for: {date}")
    # Replace with a real API call e.g. Google Calendar API
    return f"On {date} you have: 9am Standup, 2pm Product Review"

# ════════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS — manually written JSON (LangChain generates these for you)
# ════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar_events",
            "description": "Get calendar events for a specific date",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format"
                    }
                },
                "required": ["date"]
            }
        }
    }
]

# ════════════════════════════════════════════════════════════════════════
# FUNCTION MAP — manually routes tool name → callable
# (AgentExecutor does this automatically in LangChain)
# ════════════════════════════════════════════════════════════════════════

FUNCTION_MAP = {
    "get_weather": get_weather,
    "get_calendar_events": get_calendar_events,
}

# ════════════════════════════════════════════════════════════════════════
# MEMORY — simple list we manage ourselves
# (ConversationBufferMemory handles this in LangChain)
# ════════════════════════════════════════════════════════════════════════

chat_history = []

# ════════════════════════════════════════════════════════════════════════
# AGENT LOOP — the entire while True block that LangChain's
# AgentExecutor replaces
# ════════════════════════════════════════════════════════════════════════

MAX_ITERATIONS = 5          # Stop if LLM loops more than this
AGENT_TIMEOUT  = 30         # Wall-clock limit for the entire run (seconds)

def run_agent(user_input: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")

    # Build messages: system + full history + new user message
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a helpful assistant with access to weather and calendar tools. "
                f"Today's date is {today}. If a tool returns an error, explain it to the "
                f"user and suggest alternatives."
            )
        },
        *chat_history,                          # inject memory manually
        {"role": "user", "content": user_input}
    ]

    iterations = 0
    start_time = datetime.now()

    while True:
        # ── Max iteration guard ───────────────────────────────────────────
        if iterations >= MAX_ITERATIONS:
            logger.warning("Max iterations reached — stopping agent loop.")
            return "[Agent stopped] Maximum number of tool calls reached."

        # ── Wall-clock timeout guard ─────────────────────────────────────
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed > AGENT_TIMEOUT:
            logger.warning(f"Agent timeout after {elapsed:.1f}s")
            return "[Agent stopped] Execution time limit exceeded."

        iterations += 1
        logger.info(f"Agent iteration {iterations}/{MAX_ITERATIONS}")

        # ── STREAMING — manually collect chunks ──────────────────────────
        print("\nAssistant: ", end="", flush=True)

        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            stream=True
        )

        # Accumulate streamed content
        full_content    = ""
        tool_calls_acc  = {}   # index → {id, name, arguments}

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            # Stream text tokens live
            if delta.content:
                print(delta.content, end="", flush=True)
                full_content += delta.content

            # Accumulate tool call fragments (streamed in pieces)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name or "" if tc.function else "",
                            "arguments": ""
                        }
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments

        print()  # newline after streamed output

        tool_calls = list(tool_calls_acc.values())

        # ── No tool calls — LLM has a final answer ────────────────────────
        if not tool_calls:
            # Save turn to memory manually
            chat_history.append({"role": "user",      "content": user_input})
            chat_history.append({"role": "assistant",  "content": full_content})
            return full_content

        # ── Tool calls requested — build assistant message ────────────────
        assistant_msg = {
            "role": "assistant",
            "content": full_content or None,
            "tool_calls": [
                {
                    "id":       tc["id"],
                    "type":     "function",
                    "function": {
                        "name":      tc["name"],
                        "arguments": tc["arguments"]
                    }
                }
                for tc in tool_calls
            ]
        }
        messages.append(assistant_msg)

        # ── Execute each tool and inject result back into messages ─────────
        for tc in tool_calls:
            fn_name = tc["name"]
            try:
                fn_args = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                fn_args = {}

            logger.info(f"Tool call → {fn_name}({fn_args})")

            if fn_name in FUNCTION_MAP:
                result = FUNCTION_MAP[fn_name](**fn_args)
            else:
                result = f"[ERROR] Unknown tool: {fn_name}"

            logger.info(f"Tool result ← {result}")

            # Inject result — must match tool_call_id
            messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      str(result)
            })

        # Loop again — LLM will now read tool results and decide next step

# ════════════════════════════════════════════════════════════════════════
# MULTI-TURN DEMO
# ════════════════════════════════════════════════════════════════════════

def chat(user_input: str):
    print(f"\n{'─' * 60}")
    print(f"You: {user_input}")
    print(f"{'─' * 60}")
    run_agent(user_input)

if __name__ == "__main__":
    # Turn 1 — uses both tools
    chat("What's the weather in Tampa and what meetings do I have today?")

    # Turn 2 — memory means it knows "today" and "Tampa" from Turn 1
    chat("Is it a good day to hold the 9am meeting outside?")

    # Turn 3 — triggers error recovery (bad date format)
    chat("What's on my calendar for the 25th of April?")

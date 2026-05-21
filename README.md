# AI Agent Project — With vs Without LangChain

Two implementations of the same Weather + Calendar agent showing what
LangChain abstracts away vs what you manage yourself.

---

## Project Structure

```
agent_project/
├── requirements.txt
├── .env.example
├── README.md
│
├── without_langchain/
│   └── agent.py        ← Raw OpenAI API, everything manual
│
└── with_langchain/
    └── agent.py        ← LangChain handles the heavy lifting
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=your-key-here
```

---

## Run

```bash
# Without LangChain
python without_langchain/agent.py

# With LangChain
python with_langchain/agent.py
```

Both agents run the same 3-turn conversation:
1. Weather in Tampa + today's meetings (uses both tools)
2. Should I hold the meeting outside? (uses memory from turn 1)
3. What's on my calendar for the 25th of April? (triggers error recovery)

---

## Features — Where Each Lives

| Feature              | without_langchain/agent.py              | with_langchain/agent.py              |
|----------------------|-----------------------------------------|--------------------------------------|
| Tool schema          | Handwritten JSON dict                   | `StructuredTool` + Pydantic          |
| Tool routing         | `FUNCTION_MAP` dict + manual lookup     | `AgentExecutor` (automatic)          |
| Agent loop           | Manual `while True` block               | `AgentExecutor` (automatic)          |
| Streaming            | Manual chunk accumulation loop          | `streaming=True` + callback          |
| Timeout (tool)       | `@with_timeout` decorator               | `@with_timeout` decorator (same)     |
| Timeout (agent)      | Manual `elapsed > AGENT_TIMEOUT` check  | `max_execution_time=30`              |
| Max iterations       | Manual `iterations >= MAX_ITERATIONS`   | `max_iterations=5`                   |
| Error recovery (tool)| `@with_error_recovery` decorator        | `@with_error_recovery` decorator     |
| Error recovery (LLM) | `try/except json.JSONDecodeError`       | `handle_parsing_errors=...`          |
| Memory               | Manual `chat_history` list              | `ConversationBufferMemory`           |

---

## Swapping in Real APIs

Both files have clearly marked comments where mock responses can be
replaced with real API calls:

**Weather** — replace the return string in `get_weather()` with:
```python
import requests
resp = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=YOUR_KEY")
return resp.json()["weather"][0]["description"]
```

**Calendar** — replace the return string in `get_calendar_events()` with:
```python
# Google Calendar API call using google-auth + googleapiclient
```

# INDEX.md — Key Terms & Concepts to Learn

This index covers every important term, concept, library, and pattern used across all four projects — organized by project and topic. Use this as a study guide alongside the code.

---

## Universal Concepts (apply to all projects)

### LLM & API Fundamentals

**OpenAI Chat Completions API**
The core API for talking to GPT models. You send a list of messages (system, user, assistant, tool) and receive a completion. Understanding the message format is foundational — all frameworks are just wrappers around this.
*See: `without_langchain/agent.py` → `client.chat.completions.create()`*

**System Message**
A special message (role: `"system"`) at the start of the conversation that sets the assistant's behavior, persona, and context (like today's date). Injected on every call.

**Tool / Function Calling**
The mechanism that lets an LLM request that your code run a function. The LLM returns a structured `tool_calls` object instead of plain text. Your code executes the function, then feeds the result back as a `tool` message. The LLM reads it and continues.

**Tool Schema / JSON Schema**
A structured description of a tool — its name, what it does, and what parameters it accepts. The LLM reads these to know what it can call. LangChain and LangGraph generate these automatically from Python type hints; in Project 1, you write them by hand.

**Multi-Turn Conversation**
A conversation with memory across turns. Turn 2 can reference facts from Turn 1. Requires maintaining a message history that grows with each exchange.

**Streaming**
Instead of waiting for the full response, the LLM sends tokens as they're generated. Useful for UX — the user sees output immediately. Requires chunk accumulation logic to reassemble the full response for tool call parsing.

**Temperature**
Controls LLM output randomness. `0` = deterministic (same input → same output every time). Best for agents that call tools, since you want consistent, reliable decisions.

---

### Python Patterns

**Decorator Pattern (`@`)**
A function that wraps another function to add behavior. Projects 1 & 2 use two custom decorators:
- `@with_timeout(seconds)` — kills the function if it runs too long (uses `signal.SIGALRM`, Unix/macOS only)
- `@with_error_recovery` — catches all exceptions and returns them as readable strings so the LLM can adapt instead of crashing

**`functools.wraps`**
Used inside decorators to preserve the original function's name and docstring. Without it, all decorated functions would appear to be named `wrapper`.

**`signal.SIGALRM`**
A Unix OS signal used to implement timeouts. `signal.alarm(n)` schedules a `SIGALRM` after `n` seconds. When it fires, a custom handler raises a `ToolTimeoutError`. Only works on macOS/Linux.

**`asyncio.run()`**
Runs an async coroutine synchronously. Used in `with_mcp/` as a bridge between the async MCP client and the sync `StructuredTool.func` that LangChain expects.

**Abstract Base Class (ABC)**
`BaseMCPClient` inherits from `ABC` (abstract base class) and `IMCPClient` (the interface). Forces subclasses to implement specific methods. Used in `with_mcp/` to create a reusable MCP client pattern.

**`@abstractmethod`**
Marks a method in an ABC that subclasses must implement. Provides a contract — if a subclass doesn't implement `list_tools()` or `call_tool()`, Python raises an error at class definition time.

---

## Project 1: `without_langchain/` — Key Terms

**`FUNCTION_MAP`**
A plain Python dict mapping tool name strings to callable functions: `{"get_weather": get_weather}`. When the LLM requests a tool call by name, you look it up here and call it. LangChain and LangGraph replace this with automatic routing.

**Agent Loop**
The `while True:` block that runs repeatedly: send messages → get response → if tool calls: execute tools and loop again → if no tool calls: return answer. This is what `AgentExecutor` and `StateGraph` abstract away.

**Manual Chunk Accumulation**
In streaming mode, tool call arguments arrive in fragments across multiple chunks. You must manually concatenate `tc.function.arguments` fragments into a full JSON string before you can parse it. `tool_calls_acc` (a dict keyed by chunk index) does this.

**`chat_history`**
A plain Python list of message dicts that grows with each turn. Manually prepended to every new call so the LLM has context. This is the raw form of "memory."

**`MAX_ITERATIONS` / `AGENT_TIMEOUT`**
Manual safety guards. `MAX_ITERATIONS` stops infinite tool-call loops. `AGENT_TIMEOUT` stops the agent if too much wall-clock time passes. Both are implemented with manual checks inside the loop.

**`ToolTimeoutError`**
A custom exception class raised by `@with_timeout` when a tool function exceeds its time limit. Caught by `@with_error_recovery` and converted to a string the LLM can read.

---

## Project 2: `with_langchain/` — Key Terms

**LangChain**
An open-source framework for building LLM applications. Provides abstractions for prompts, LLMs, tools, memory, and agent loops. Replaces manual boilerplate with configurable components.

**`StructuredTool`**
A LangChain class that wraps a Python function as an agent tool. Takes a `func`, a `description`, and an `args_schema` (Pydantic model). Automatically generates the JSON schema the LLM needs.
*See: `with_langchain/tools/weather.py`*

**Pydantic `BaseModel`**
A Python library for data validation using type annotations. In LangChain, Pydantic models define tool input schemas. LangChain reads the field names, types, and `Field(description=...)` values to auto-generate JSON schema for the LLM.

**`Field(description=...)`**
A Pydantic utility that adds metadata to a model field. The `description` is included in the JSON schema sent to the LLM, helping it understand what value to pass.

**`AgentExecutor`**
LangChain's built-in agent loop. Takes an `agent` (the LLM + prompt), `tools` (list of StructuredTools), and `memory`. Runs the tool-call loop automatically, handles max iterations, timeouts, and parsing errors. Replaces the manual `while True:` from Project 1.

**`create_tool_calling_agent`**
A LangChain factory function that wires together an LLM, a list of tools, and a prompt template into a runnable "agent" object. This agent is then passed to `AgentExecutor`.

**`ConversationBufferMemory`**
LangChain's in-memory conversation store. Saves input/output pairs from each turn and injects them into the prompt as `chat_history`. Replaces the manual `chat_history = []` list.

**`ChatPromptTemplate`**
A structured prompt definition with placeholders for system message, memory (`chat_history`), user input, and agent scratchpad. The scratchpad is where tool call/result pairs are injected during the loop.

**`MessagesPlaceholder`**
A special slot in a `ChatPromptTemplate` that gets filled with a list of messages at runtime. Used for `chat_history` and `agent_scratchpad`.

**`StreamingStdOutCallbackHandler`**
A LangChain callback that prints each streamed token to stdout as it arrives. Passed to `ChatOpenAI(callbacks=[...])`. Replaces the manual streaming loop from Project 1.

**`ChatOpenAI`**
LangChain's wrapper around OpenAI's chat models. Accepts model name, temperature, streaming flag, and callbacks. Handles the raw API call internally.

**`return_intermediate_steps=True`**
An `AgentExecutor` option that includes the list of (action, observation) pairs in the result dict. Lets you inspect which tools were called and what they returned.

**`handle_parsing_errors`**
An `AgentExecutor` option that gracefully handles cases where the LLM returns malformed output. Instead of crashing, it sends the error message back to the LLM to try again.

---

## Project 3: `with_langgraph/` — Key Terms

**LangGraph**
A library (built on LangChain) for building **stateful, graph-based AI agents**. Instead of a hidden loop, you define explicit nodes (computations) and edges (transitions). Enables multi-agent orchestration, human-in-the-loop, and inspectable state.

**`StateGraph`**
The main LangGraph builder class. You add nodes and edges to it, then call `.compile()` to get a runnable graph. Think of it as a blueprint for your agent's execution flow.

**`AgentState` / `TypedDict`**
The shared data container that flows through every node. Defined as a Python `TypedDict` so all fields are type-checked. Every node reads from it and returns a partial update (only changed fields).

**`Annotated[list, add_messages]`**
The key pattern for the `messages` field in `AgentState`. `Annotated` attaches metadata to the type. `add_messages` is a **reducer** — instead of replacing the list, it appends new messages. This is what makes the agent loop work automatically without any manual list management.

**Reducer**
A function that defines how a state field is updated when a node returns a new value. `add_messages` is the built-in reducer for message lists. You can write custom reducers for other fields.

**Node**
A plain Python function registered in the graph with `graph.add_node("name", fn)`. Takes the full `AgentState` as input, returns a dict of partial state updates. Examples: `agent_node`, `tool_node`.

**Edge**
A connection between nodes. `add_edge("tools", "agent")` means: after `tools` runs, always go to `agent`. Unconditional.

**Conditional Edge**
A routing function that reads the current state and returns the name of the next node. `add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})` means: after `agent` runs, call `should_continue(state)` to decide where to go next.

**`should_continue`**
The routing function in `graph.py`. Checks if the last LLM message has `tool_calls`. If yes → go to `"tools"`. If no → go to `END`. This is the explicit version of what `AgentExecutor` decides internally.

**`ToolNode`**
A prebuilt LangGraph node that reads `tool_calls` from the last message, executes the matching tools, and wraps results in `ToolMessage` objects. Replaces the manual `FUNCTION_MAP` loop from Project 1 and the internal routing in `AgentExecutor`.

**`llm.bind_tools(tools)`**
Attaches the tool schemas to the LLM so it knows what it can call. Every message sent to this LLM automatically includes the tool definitions. Returns a new LLM object (`llm_with_tools`).

**`MemorySaver`**
A LangGraph checkpointer that stores a snapshot of the full graph state after every node execution, in memory. Each `thread_id` gets its own isolated history. Can be swapped for `SqliteSaver` or `RedisSaver` for real persistence.

**`thread_id`**
A key in the `configurable` config dict passed to `graph.stream()`. All messages with the same `thread_id` share a memory context. Different `thread_id`s are completely isolated — enabling parallel conversations.

**`recursion_limit`**
LangGraph's equivalent of `max_iterations`. Stops the graph if it has executed more than N nodes total. Passed in the config dict alongside `thread_id`.

**`graph.stream(input, config, stream_mode)`**
Runs the graph and yields state snapshots as events. `stream_mode="values"` gives the full state after each node. `stream_mode="updates"` gives only the delta. Replaces `executor.invoke()` + callback handlers.

**`graph.get_state(config)`**
Inspects the current state of a thread at any point. Returns the full message history, the next node that would run, and checkpoint metadata. Unique to LangGraph — there's no equivalent in LangChain.

**`set_entry_point("agent")`**
Designates which node runs first when the graph is invoked. Equivalent to `START → agent`.

**`graph.compile(checkpointer=...)`**
Locks the graph definition and returns a runnable `CompiledGraph` (also called a `Pregel` object internally). Must be called before you can invoke or stream the graph.

**Supervisor Pattern**
A multi-agent design where one **Supervisor** agent orchestrates multiple **Specialist** agents. The supervisor reads the conversation, decides which specialist should run next (using structured output), and routes accordingly. Specialists report back to the supervisor when done.

**`create_react_agent()`**
A LangGraph shorthand that builds a complete mini StateGraph (agent_node → conditional → ToolNode → agent_node cycle) for a single specialist agent. Each specialist is a full sub-graph.

**`with_structured_output(Schema)`**
Forces the LLM to always return a valid instance of a Pydantic model. Used for the supervisor's routing decision — instead of free text that needs parsing, you get a typed `SupervisorDecision(next="researcher", reasoning="...")`.

**`SupervisorState`**
Extended state that adds a `next: Optional[str]` field. The supervisor sets this field to `"researcher"`, `"analyst"`, or `"FINISH"`. The routing function reads it to decide which node runs next. Keeps routing decisions out of the messages list.

**`state_modifier`**
A string (or function) passed to `create_react_agent()` that acts as a system message for that specific specialist agent. Gives each specialist a focused role and persona without separate prompt templates.

---

## Project 4: `with_mcp/` — Key Terms

**MCP (Model Context Protocol)**
An open standard for connecting AI agents to external tools and data sources via a common protocol. Tools are external processes (servers) that accept requests over `stdio`, not just Python functions. Enables any language model to use any tool that implements the protocol.

**MCP Server**
An external executable that implements the MCP protocol. Examples in this project:
- `@openbnb/mcp-server-airbnb` (run via `npx`) — Airbnb search
- `aviationstack-mcp` (run via `uvx`) — live flight data

**`StdioServerParameters`**
Defines how to launch an MCP server subprocess: the command (`npx`, `uvx`, `python`), its arguments, and any environment variables (like API keys). The MCP client uses this to spin up the server process.

**`stdio_client`**
An async context manager from the MCP Python SDK that spawns the server subprocess and returns `(read, write)` streams. Every tool call opens a fresh connection and closes it when done.

**`ClientSession`**
The MCP session object. Wraps the `(read, write)` streams and provides high-level methods: `session.initialize()`, `session.list_tools()`, `session.call_tool()`.

**`session.initialize()`**
The MCP handshake. Must be called before any tool operations. Exchanges capability information between client and server.

**`session.list_tools()`**
Asks the MCP server what tools it offers. Returns a list of tool descriptors with name, description, and input schema. Used in `AviationstackMCPClient` to discover which flight tool to call.

**`session.call_tool(name, arguments)`**
Executes a named tool on the MCP server with a dict of arguments. Returns a result object where the actual data is in `result.content[0].text` (a JSON string).

**`npx`**
Node.js's package runner. `npx -y @openbnb/mcp-server-airbnb` downloads and runs the Airbnb MCP server without installing it permanently. The `-y` flag skips confirmation prompts.

**`uvx`**
A fast Python package runner (from the `uv` ecosystem). `uvx aviationstack-mcp` downloads and runs the Aviationstack MCP server in an isolated environment.

**`AVIATION_STACK_API_KEY`**
The Aviationstack MCP server requires a real API key for the Aviationstack flight data service. Loaded from `.env` and passed to the server subprocess via its environment variables.

**Sync Wrapper (`search_sync`)**
MCP clients are inherently async (they use `await`). But `StructuredTool.func` in LangChain must be synchronous. `search_sync` bridges this by calling `asyncio.run(self.search(...))` — it runs the async method synchronously in a new event loop.

**Interface + Base Class Pattern**
`IMCPClient` defines the contract (abstract interface). `BaseMCPClient` implements shared logic (`list_tools`, `call_tool`, `stdio` session management). `AirbnbMCPClient` and `AviationstackMCPClient` inherit from `BaseMCPClient` and only add their specific search logic. This is the **Template Method** design pattern.

**`max_execution_time=60`**
The `AgentExecutor` timeout is set to 60 seconds (vs 30 in Project 2) because MCP servers can be slow to start — they download and launch a Node or Python subprocess on the first call.

---

## Libraries & Tools Reference

| Library | What it is | Used in |
|---|---|---|
| `openai` | Official OpenAI Python SDK | All projects |
| `python-dotenv` | Loads `.env` files into environment variables | All projects |
| `langchain-core` | Core abstractions: prompts, tools, messages | Projects 2, 3, 4 |
| `langchain-openai` | LangChain's OpenAI integration (`ChatOpenAI`) | Projects 2, 3, 4 |
| `langchain-classic` | `AgentExecutor`, `ConversationBufferMemory` | Projects 2, 4 |
| `pydantic` | Data validation + schema generation | Projects 2, 4 |
| `langgraph` | Graph-based agent orchestration | Project 3 |
| `langgraph-checkpoint` | `MemorySaver`, `SqliteSaver` backends | Project 3 |
| `mcp` | MCP Python SDK (client + server) | Project 4 |
| `ruff` | Python linter + formatter (replaces flake8 + black) | Dev tooling |
| `pytest` | Test runner | Dev tooling |

---

## Concepts to Learn in Recommended Order

1. **OpenAI function/tool calling** — understand the raw message format first (Project 1 is your teacher)
2. **Python decorators** — `@with_timeout` and `@with_error_recovery` appear in Projects 1 & 2
3. **Pydantic BaseModel + Field** — how LangChain generates tool schemas (Project 2)
4. **LangChain core abstractions** — `StructuredTool`, `AgentExecutor`, `ConversationBufferMemory`, `ChatPromptTemplate` (Project 2)
5. **TypedDict + Annotated reducers** — the foundation of LangGraph state (Project 3)
6. **LangGraph StateGraph** — nodes, edges, conditional edges, compile (Project 3)
7. **MemorySaver + thread_id** — thread-based memory and state inspection (Project 3)
8. **Supervisor / multi-agent pattern** — structured output routing + sub-agents (Project 3 `supervisor.py`)
9. **MCP protocol** — stdio client/server, session, list_tools, call_tool (Project 4)
10. **Abstract base classes + interface pattern** — `IMCPClient` → `BaseMCPClient` → concrete clients (Project 4)

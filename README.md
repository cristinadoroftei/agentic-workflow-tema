# QA Agent with Tools + Prompts (ReAct Pattern)

A conversational QA agent that uses the ReAct (Reason + Act) pattern to answer questions by calling tools and reasoning about results.

## Project Structure

```
proiect/
├── agent.py                # QA agent — ReAct loop + LLM communication
├── tools/
│   ├── __init__.py         # Exports ToolWrapper
│   ├── registry.py         # TOOL_REGISTRY + @register_tool decorator
│   ├── params_models.py    # Pydantic BaseModel for each tool's parameters
│   ├── basic_tools.py      # Tool implementations: calculator, get_datetime, web_search
│   └── tool_wrapper.py     # ToolWrapper.call() + catalog()
├── prompts/
│   ├── registry.py         # PromptRegistry — loads YAML + renders with Jinja2
│   ├── planner.yaml        # Planning assistant prompt
│   ├── analyst.yaml        # Analytical assistant prompt
│   ├── summary.yaml        # Summarization assistant prompt
│   └── extract.yaml        # Extraction assistant prompt
├── .env                    # API keys (not committed)
├── .env.example            # Template for required API keys
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install pydantic httpx pyyaml jinja2 python-dotenv
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

### 3. Run the agent

```bash
cd proiect
python3 agent.py
```

## LLM Providers

The agent uses a factory pattern with automatic fallback:

| Priority | Provider | Requires |
|----------|----------|----------|
| 1 | LiteLLM proxy | Docker containers running on :4000 |
| 2 | Gemini direct | GEMINI_API_KEY in .env |
| 3 | Ollama local | Ollama running on :11434 |

If the first provider fails, the agent automatically tries the next one.

### Using with LiteLLM (Docker)

```bash
cd LLM-container-inference-script
cp ../proiect/.env .env
docker-compose up -d
```

## Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `calculator` | Evaluates math expressions | `expression: str` |
| `get_datetime` | Returns current date/time | `timezone: str` (default: UTC) |
| `web_search` | Searches the web via DuckDuckGo | `query: str`, `max_results: int` |

Tools are registered automatically using the `@register_tool` decorator. Each tool has Pydantic validation on its parameters.

## Prompts

Prompts are stored as YAML files with Jinja2 templates for dynamic variables:

```yaml
name: planner
prompt: |
  You are a {{ role }}, specialized in {{ domain }}.
  Available tools: {{ tools }}
```

Loaded and rendered at runtime via `PromptRegistry`.

## ReAct Pattern

The agent follows a Think -> Act -> Observe loop:

```
User: "What is 25 * 47 and what time is it in Bucharest?"

Iteration 1 (Think + Act):
  -> calculator(expression="25 * 47")        -> 1175
  -> get_datetime(timezone="Europe/Bucharest") -> 2026-05-25 00:28 EEST

Iteration 2 (Final answer):
  -> "25 * 47 = 1175. The time in Bucharest is 00:28 EEST."
```

Features:
- `max_iterations` safety limit (default: 5)
- Error handling — tool errors are returned to the LLM as text, not crashes
- Multiple tool calls per iteration
- Session history — follow-up questions have full conversation context

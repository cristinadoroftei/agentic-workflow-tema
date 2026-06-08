import json
import os

import anthropic
import httpx
from dotenv import load_dotenv

import tools.basic_tools  # triggers @register_tool for all tools
from tools import ToolWrapper
from prompts.registry import get_prompt_registry

# --- Configuration ---
load_dotenv()
MAX_ITERATIONS = 10

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GEMINI_MODEL = "gemini-2.5-flash"

# Default provider with fallback order
PROVIDER_ORDER = ["claude", "gemini"]


# --- Sub-step A: LLM Communication ---

def _build_claude_tools(tool_catalog: list[dict]) -> list[dict]:
    """Convert our catalog to Claude tool format."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["input_schema"],
        }
        for t in tool_catalog
    ]


def _call_claude(system_prompt: str, messages: list[dict], tool_catalog: list[dict]) -> dict:
    """Call Claude natively — no conversion needed."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system_prompt,
        tools=_build_claude_tools(tool_catalog),
        messages=messages,
    )
    # Serialize every block losslessly to plain dicts. model_dump() preserves all
    # fields (and any future block types — thinking, server_tool_use, ...) so we
    # never silently drop content the API sent back. exclude_unset keeps the dicts
    # minimal and round-trip-safe when sent back to the API.
    content = [block.model_dump(exclude_unset=True) for block in response.content]
    return {"role": "assistant", "content": content}


def _messages_to_openai(system_prompt: str, messages: list[dict]) -> list[dict]:
    """Convert Claude-format messages to OpenAI format for Gemini fallback."""
    openai_messages = [{"role": "system", "content": system_prompt}]

    for msg in messages:
        if msg["role"] == "user":
            if isinstance(msg["content"], str):
                openai_messages.append(msg)
            elif isinstance(msg["content"], list):
                # Tool results — each becomes a separate "tool" message in OpenAI
                for item in msg["content"]:
                    if item["type"] == "tool_result":
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": item["tool_use_id"],
                            "content": item["content"],
                        })

        elif msg["role"] == "assistant":
            content = None
            tool_calls = []
            for block in msg["content"]:
                if block["type"] == "text":
                    content = block["text"]
                elif block["type"] == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    })
            result = {"role": "assistant", "content": content}
            if tool_calls:
                result["tool_calls"] = tool_calls
            openai_messages.append(result)

    return openai_messages


def _call_gemini(system_prompt: str, messages: list[dict], tool_catalog: list[dict]) -> dict:
    """Call Gemini, converting Claude-format messages to OpenAI format."""
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tool_catalog
    ]
    openai_messages = _messages_to_openai(system_prompt, messages)

    response = httpx.post(
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('GEMINI_API_KEY')}"},
        json={
            "model": GEMINI_MODEL,
            "messages": openai_messages,
            "tools": openai_tools,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    openai_msg = response.json()["choices"][0]["message"]

    # Convert Gemini's OpenAI-format response back to Claude format
    content = []
    if openai_msg.get("content"):
        content.append({"type": "text", "text": openai_msg["content"]})
    for tc in openai_msg.get("tool_calls", []):
        content.append({
            "type": "tool_use",
            "id": tc["id"],
            "name": tc["function"]["name"],
            "input": json.loads(tc["function"]["arguments"]),
        })
    return {"role": "assistant", "content": content}


PROVIDERS = {
    "claude": _call_claude,
    "gemini": _call_gemini,
}


def call_llm(system_prompt: str, messages: list[dict], tool_catalog: list[dict]) -> dict:
    """Try each provider in order until one succeeds."""
    for provider_name in PROVIDER_ORDER:
        try:
            result = PROVIDERS[provider_name](system_prompt, messages, tool_catalog)
            print(f"  [Using: {provider_name}]")
            return result
        except Exception as e:
            print(f"  [Provider {provider_name} failed: {e}]")

    raise RuntimeError("All LLM providers failed.")


# --- Sub-step B: ReAct Loop ---

def react_loop(system_prompt: str, messages: list[dict]) -> str:
    """The Think -> Act -> Observe loop. Returns the final answer.

    Uses Claude's native message format:
    - Assistant content = list of blocks (text / tool_use)
    - Tool results = user message with tool_result blocks
    """
    tool_catalog = ToolWrapper.catalog()

    for i in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {i + 1} ---")

        # THINK: ask the LLM what to do
        try:
            assistant_msg = call_llm(system_prompt, messages, tool_catalog)
        except RuntimeError as e:
            return f"Error: {e}. Please wait a moment and try again (rate limit)."
        messages.append(assistant_msg)

        # Split content blocks into text and tool_use
        tool_use_blocks = [b for b in assistant_msg["content"] if b["type"] == "tool_use"]
        text_blocks = [b for b in assistant_msg["content"] if b["type"] == "text"]

        # If no tool calls -> the LLM has a final answer
        if not tool_use_blocks:
            print("LLM returned final answer.")
            return text_blocks[0]["text"] if text_blocks else "No response."

        # ACT: execute each tool and collect results
        tool_results = []
        for block in tool_use_blocks:
            print(f"  Tool call: {block['name']}({block['input']})")
            result = ToolWrapper.call(block["name"], block["input"])
            print(f"  Result: {result}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": result,
            })

        # OBSERVE: send all results back as one user message
        messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached without a final answer."


# --- Sub-step C: Main entry point ---

# Default Jinja2 variables per prompt template
PROMPT_DEFAULTS = {
    "planner": {"role": "planning assistant", "domain": "general knowledge", "max_words": 300},
    "analyst": {"role": "data analyst", "domain": "research and analysis"},
    "summary": {"role": "summarization expert", "domain": "concise communication", "max_words": 150},
    "extract": {"role": "information extraction specialist", "domain": "data extraction", "format": "bullet points"},
}


def build_system_prompt(registry, tool_names: str, prompt_name: str) -> str:
    """Render a system prompt by name with its default variables."""
    defaults = PROMPT_DEFAULTS[prompt_name]
    return registry.render(prompt_name, tools=tool_names, **defaults)


def main():
    registry = get_prompt_registry()
    tool_names = ", ".join(t["name"] for t in ToolWrapper.catalog())
    available = registry.list_templates()

    # Let the user pick a starting mode
    print("Available modes:", ", ".join(available))
    print("Commands: /mode <name> to switch, /modes to list, exit to quit\n")

    current_mode = "planner"
    system_prompt = build_system_prompt(registry, tool_names, current_mode)
    messages = []
    print(f"[Mode: {current_mode}]")

    # Interactive chat loop
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            break

        # Command: list available modes
        if question == "/modes":
            print(f"Available: {', '.join(available)}  (current: {current_mode})")
            continue

        # Command: switch mode
        if question.startswith("/mode "):
            new_mode = question.split(" ", 1)[1].strip()
            if new_mode not in available:
                print(f"Unknown mode '{new_mode}'. Available: {', '.join(available)}")
                continue
            current_mode = new_mode
            system_prompt = build_system_prompt(registry, tool_names, current_mode)
            messages = []
            print(f"[Switched to: {current_mode}] (conversation reset)")
            continue

        messages.append({"role": "user", "content": question})
        answer = react_loop(system_prompt, messages)
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()

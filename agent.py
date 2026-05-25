import json
import os

import httpx
from dotenv import load_dotenv

import tools.basic_tools  # triggers @register_tool for all tools
from tools import ToolWrapper
from prompts.registry import get_prompt_registry

# --- Configuration ---
load_dotenv()
MAX_ITERATIONS = 5

# Provider registry — factory pattern
# Each provider has: base_url, model, headers builder, and how to build the request body
PROVIDERS = {
    "litellm": {
        "base_url": "http://localhost:4000/v1",
        "model": "gemini",
        "headers": lambda: {},
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-flash",
        "headers": lambda: {"Authorization": f"Bearer {os.getenv('GEMINI_API_KEY')}"},
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3",
        "headers": lambda: {},
    },
}

# Default provider with fallback order
PROVIDER_ORDER = ["litellm", "gemini", "ollama"]


# --- Sub-step A: LLM Communication ---

def _build_openai_tools(tool_catalog: list[dict]) -> list[dict]:
    """Convert our catalog to OpenAI-compatible tool format."""
    return [
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


def _call_provider(provider_name: str, messages: list[dict], openai_tools: list[dict]) -> dict:
    """Call a specific LLM provider. Raises on failure."""
    provider = PROVIDERS[provider_name]
    response = httpx.post(
        f"{provider['base_url']}/chat/completions",
        headers=provider["headers"](),
        json={
            "model": provider["model"],
            "messages": messages,
            "tools": openai_tools,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


def call_llm(messages: list[dict], tool_catalog: list[dict]) -> dict:
    """Try each provider in order until one succeeds."""
    openai_tools = _build_openai_tools(tool_catalog)

    for provider_name in PROVIDER_ORDER:
        try:
            result = _call_provider(provider_name, messages, openai_tools)
            print(f"  [Using: {provider_name}]")
            return result
        except Exception as e:
            print(f"  [Provider {provider_name} failed: {e}]")

    raise RuntimeError("All LLM providers failed.")


# --- Sub-step B: ReAct Loop ---

def react_loop(messages: list[dict]) -> str:
    """The Think -> Act -> Observe loop. Returns the final answer."""
    tool_catalog = ToolWrapper.catalog()

    for i in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {i + 1} ---")

        # THINK: ask the LLM what to do
        assistant_msg = call_llm(messages, tool_catalog)
        messages.append(assistant_msg)

        # If no tool calls -> the LLM has a final answer
        if not assistant_msg.get("tool_calls"):
            print("LLM returned final answer.")
            return assistant_msg.get("content", "No response.")

        # ACT: execute each tool the LLM requested
        for tool_call in assistant_msg["tool_calls"]:
            func = tool_call["function"]
            tool_name = func["name"]
            tool_args = json.loads(func["arguments"])

            print(f"  Tool call: {tool_name}({tool_args})")
            result = ToolWrapper.call(tool_name, tool_args)
            print(f"  Result: {result}")

            # OBSERVE: send the result back to the LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result,
            })

    return "Max iterations reached without a final answer."


# --- Sub-step C: Main entry point ---

def main():
    # Load the planner prompt and render it with our tools
    registry = get_prompt_registry()
    tool_names = ", ".join(t["name"] for t in ToolWrapper.catalog())

    system_prompt = registry.render(
        "planner",
        role="QA assistant",
        domain="general knowledge",
        tools=tool_names,
        max_words=300,
    )

    print("System prompt:")
    print(system_prompt)
    print("=" * 50)

    # Session messages — persists across questions so the agent has context
    messages = [{"role": "system", "content": system_prompt}]

    # Interactive chat loop
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": question})
        answer = react_loop(messages)
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()

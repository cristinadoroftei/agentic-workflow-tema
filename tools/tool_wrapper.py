from tools.registry import TOOL_REGISTRY


class ToolWrapper:
    @staticmethod
    def call(name: str, args: dict) -> str:
        """Look up a tool by name, validate args with Pydantic, execute it."""
        # 1. Lookup — does this tool exist?
        if name not in TOOL_REGISTRY:
            return f"Error: tool '{name}' does not exist."

        tool = TOOL_REGISTRY[name]

        # 2. Validate — Pydantic checks types and constraints
        try:
            params = tool["params_model"](**args)
        except Exception as e:
            return f"Validation error for '{name}': {e}"

        # 3. Execute and return the result
        try:
            return str(tool["func"](params))
        except Exception as e:
            return f"Execution error in '{name}': {e}"

    @staticmethod
    def catalog() -> list[dict]:
        """Generate the tool catalog for the LLM (JSON Schema format)."""
        return [
            {
                "name": name,
                "description": tool["description"],
                "input_schema": tool["params_model"].model_json_schema(),
            }
            for name, tool in TOOL_REGISTRY.items()
        ]

from inspect import Parameter


import inspect
from pydantic import BaseModel

# Global dictionary: tool_name -> {func, params_model, description}
TOOL_REGISTRY: dict[str, dict] = {}


def register_tool(func):
    """Decorator that registers a function as a tool in TOOL_REGISTRY.

    Rules enforced:
    - The function must accept exactly one parameter of type BaseModel
    - The function must have a docstring (it becomes the tool description for the LLM)
    - The docstring must be at least 15 characters (so the LLM has enough context)
    """
    sig = inspect.signature(func)
    params = list[Parameter](sig.parameters.values())

    # Rule 1: exactly one parameter, and it must be a Pydantic BaseModel
    if len(params) != 1 or not issubclass(params[0].annotation, BaseModel):
        raise TypeError(
            f"{func.__name__}: must have exactly one parameter of type BaseModel"
        )

    # Rule 2: docstring is mandatory (it becomes the description the LLM sees)
    docstring = (func.__doc__ or "").strip()
    if not docstring:
        raise ValueError(
            f"{func.__name__}: docstring is required — it becomes the "
            f"tool description visible to the LLM."
        )

    # Rule 3: docstring must be descriptive enough
    if len(docstring) < 15:
        raise ValueError(
            f"{func.__name__}: docstring too short ({len(docstring)} chars). "
            f"The LLM needs at least 15 characters to decide when to use this tool."
        )

    # All checks passed — register the tool
    TOOL_REGISTRY[func.__name__] = {
        "func": func,
        "params_model": params[0].annotation,
        "description": docstring,
    }

    return func

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from jinja2 import Template


@dataclass(frozen=True)
class PromptTemplate:
    """Internal representation of a YAML prompt file."""
    name: str
    version: str
    prompt: str
    description: str = ""


class PromptRegistry:
    """Loads all YAML prompt files from a folder and renders them with Jinja2."""

    def __init__(self, folder: str):
        self._templates = self._load(folder)

    def _load(self, folder: str) -> dict[str, PromptTemplate]:
        """Read all .yaml files from the folder into a dict {name: PromptTemplate}."""
        templates = {}
        for path in Path(folder).rglob("*.yaml"):
            data = yaml.safe_load(path.read_text())
            tpl = PromptTemplate(**data)
            templates[tpl.name] = tpl
        return templates

    def render(self, name: str, **variables) -> str:
        """Render a prompt template by name, filling in Jinja2 variables."""
        template = self._templates[name]
        return Template(template.prompt).render(**variables)

    def list_templates(self) -> list[str]:
        """Return all available template names."""
        return list[str](self._templates.keys())


@lru_cache(maxsize=1)
def get_prompt_registry() -> PromptRegistry:
    """Singleton: one load, reused everywhere."""
    return PromptRegistry(folder="prompts/")

"""Prompt template engine for video and image captioning.

Templates use Python's str.format() syntax for variable substitution.
Variables are auto-extracted from {placeholder} patterns in the template string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_VAR_PATTERN = re.compile(r"\{(\w+)\}")


def _extract_variables(template: str) -> list[str]:
    """Extract unique variable names from a template string, preserving order."""
    seen: set[str] = set()
    variables: list[str] = []
    for match in _VAR_PATTERN.finditer(template):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            variables.append(name)
    return variables


@dataclass(frozen=True)
class PromptTemplate:
    """A captioning prompt with optional variable placeholders.

    Variables are auto-extracted from {placeholder} patterns in the template.
    """

    name: str
    template: str
    mode: str  # "video" | "image" | "both"
    category: str
    variables: list[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", _extract_variables(self.template))


BUILT_IN_TEMPLATES: dict[str, PromptTemplate] = {
    "general": PromptTemplate(
        name="general",
        template="Describe what happens in this {media_type}.",
        mode="both",
        category="general",
    ),
    "detailed": PromptTemplate(
        name="detailed",
        template=(
            "Describe this {media_type} in detail, including characters, "
            "setting, actions, and atmosphere."
        ),
        mode="both",
        category="general",
    ),
    "motion": PromptTemplate(
        name="motion",
        template=(
            "Describe the movement and actions in this video. Focus on what moves, "
            "how it moves, and any changes in motion."
        ),
        mode="video",
        category="motion",
    ),
    "composition": PromptTemplate(
        name="composition",
        template=(
            "Describe the visual composition: framing, camera angle, "
            "camera movement, lighting, and color palette."
        ),
        mode="both",
        category="composition",
    ),
    "character": PromptTemplate(
        name="character",
        template=(
            "Describe the characters: appearance, clothing, expressions, and actions. "
            "Refer to the main character as {character_name}."
        ),
        mode="both",
        category="character",
    ),
    "narrative": PromptTemplate(
        name="narrative",
        template=(
            "Describe this scene as if writing a screenplay. "
            "Include setting, character actions, and emotional tone."
        ),
        mode="both",
        category="narrative",
    ),
}


def get_template(name: str) -> PromptTemplate:
    """Get a built-in template by name. Raises KeyError if not found."""
    return BUILT_IN_TEMPLATES[name]


def get_templates(
    mode: str | None = None,
    category: str | None = None,
) -> list[PromptTemplate]:
    """List built-in templates, optionally filtered by mode and/or category."""
    templates = list(BUILT_IN_TEMPLATES.values())
    if mode is not None:
        templates = [t for t in templates if t.mode in (mode, "both")]
    if category is not None:
        templates = [t for t in templates if t.category == category]
    return templates


def render_template(
    template: PromptTemplate,
    variables: dict[str, str],
    media_type: str = "video",
) -> str:
    """Render a template with variable substitution.

    Args:
        template: The prompt template to render.
        variables: Key-value pairs for variable substitution.
        media_type: Auto-fill value for {media_type} if not in variables.

    Returns:
        Rendered prompt string.

    Raises:
        KeyError: If a required variable is missing from variables.
    """
    merged = {"media_type": media_type}
    merged.update(variables)  # Explicit variables override auto-fill
    return template.template.format(**merged)

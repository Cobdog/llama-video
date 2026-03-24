"""Tests for prompt template engine."""

from __future__ import annotations

import pytest

from llama_video.templates import (
    BUILT_IN_TEMPLATES,
    PromptTemplate,
    get_template,
    get_templates,
    render_template,
)


class TestPromptTemplate:
    """Test PromptTemplate dataclass."""

    def test_variables_auto_extracted(self):
        t = PromptTemplate(
            name="test",
            template="Describe {character_name} in this {media_type}.",
            mode="both",
            category="general",
        )
        assert t.variables == ["character_name", "media_type"]

    def test_no_variables(self):
        t = PromptTemplate(
            name="test",
            template="Describe this video.",
            mode="video",
            category="general",
        )
        assert t.variables == []

    def test_duplicate_variables_deduplicated(self):
        t = PromptTemplate(
            name="test",
            template="{foo} and {bar} and {foo} again.",
            mode="both",
            category="general",
        )
        assert t.variables == ["foo", "bar"]

    def test_frozen(self):
        t = PromptTemplate(
            name="test",
            template="test",
            mode="both",
            category="general",
        )
        with pytest.raises(AttributeError):
            t.name = "changed"  # type: ignore[misc]


class TestBuiltInTemplates:
    """Test built-in template registry."""

    def test_general_template_exists(self):
        assert "general" in BUILT_IN_TEMPLATES

    def test_detailed_template_exists(self):
        assert "detailed" in BUILT_IN_TEMPLATES

    def test_motion_template_is_video_only(self):
        t = BUILT_IN_TEMPLATES["motion"]
        assert t.mode == "video"

    def test_composition_template_is_both(self):
        t = BUILT_IN_TEMPLATES["composition"]
        assert t.mode == "both"

    def test_character_template_has_character_name_variable(self):
        t = BUILT_IN_TEMPLATES["character"]
        assert "character_name" in t.variables

    def test_all_templates_have_required_fields(self):
        for name, t in BUILT_IN_TEMPLATES.items():
            assert t.name == name
            assert t.mode in ("video", "image", "both")
            assert len(t.template) > 0
            assert len(t.category) > 0


class TestGetTemplate:
    """Test template retrieval."""

    def test_get_existing_template(self):
        t = get_template("general")
        assert t.name == "general"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get_template("nonexistent")


class TestGetTemplates:
    """Test filtered template listing."""

    def test_get_all_templates(self):
        templates = get_templates()
        assert len(templates) >= 6

    def test_filter_by_video_mode(self):
        templates = get_templates(mode="video")
        for t in templates:
            assert t.mode in ("video", "both")

    def test_filter_by_image_mode(self):
        templates = get_templates(mode="image")
        for t in templates:
            assert t.mode in ("image", "both")

    def test_filter_by_category(self):
        templates = get_templates(category="general")
        for t in templates:
            assert t.category == "general"


class TestRenderTemplate:
    """Test template rendering with variable substitution."""

    def test_render_with_variables(self):
        t = PromptTemplate(
            name="test",
            template="Describe {character_name} in this {media_type}.",
            mode="both",
            category="general",
        )
        result = render_template(t, {"character_name": "Kiki", "media_type": "video"})
        assert result == "Describe Kiki in this video."

    def test_render_with_missing_variable_raises(self):
        t = PromptTemplate(
            name="test",
            template="Focus on {focus_area}.",
            mode="both",
            category="general",
        )
        with pytest.raises(KeyError):
            render_template(t, {})

    def test_render_with_no_variables(self):
        t = PromptTemplate(
            name="test",
            template="Describe this video.",
            mode="video",
            category="general",
        )
        result = render_template(t, {})
        assert result == "Describe this video."

    def test_render_with_extra_variables_ignores_them(self):
        t = PromptTemplate(
            name="test",
            template="Describe this video.",
            mode="video",
            category="general",
        )
        result = render_template(t, {"unused": "value"})
        assert result == "Describe this video."

    def test_render_media_type_auto_fills(self):
        t = PromptTemplate(
            name="test",
            template="Describe this {media_type}.",
            mode="both",
            category="general",
        )
        result = render_template(t, {}, media_type="image")
        assert result == "Describe this image."

    def test_render_media_type_explicit_overrides_auto(self):
        t = PromptTemplate(
            name="test",
            template="Describe this {media_type}.",
            mode="both",
            category="general",
        )
        result = render_template(t, {"media_type": "clip"}, media_type="video")
        assert result == "Describe this clip."

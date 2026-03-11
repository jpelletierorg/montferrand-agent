"""Tests for agent.py — env resolution, prompt rendering, and fallback pricing.

All tests are pure logic — no LLM calls.
"""

import pytest

from montferrand_agent.agent import (
    DEMO_TENANT_PROFILE,
    MASTER_PROMPT_TEMPLATE,
    _FALLBACK_PRICING,
    _require_env,
    _resolve_env,
    get_fallback_pricing,
    render_prompt,
)


# ---------------------------------------------------------------------------
# _resolve_env
# ---------------------------------------------------------------------------


class TestResolveEnv:
    def test_returns_env_var_when_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEST_VAR_A", "value_a")
        assert _resolve_env("TEST_VAR_A", default="fallback") == "value_a"

    def test_skips_empty_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEST_VAR_A", "")
        monkeypatch.setenv("TEST_VAR_B", "value_b")
        assert _resolve_env("TEST_VAR_A", "TEST_VAR_B", default="fallback") == "value_b"

    def test_skips_whitespace_only(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEST_VAR_A", "   ")
        assert _resolve_env("TEST_VAR_A", default="fallback") == "fallback"

    def test_returns_default_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TEST_VAR_NONEXISTENT", raising=False)
        assert _resolve_env("TEST_VAR_NONEXISTENT", default="fallback") == "fallback"

    def test_returns_first_non_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEST_VAR_A", "")
        monkeypatch.setenv("TEST_VAR_B", "  ")
        monkeypatch.setenv("TEST_VAR_C", "winner")
        result = _resolve_env("TEST_VAR_A", "TEST_VAR_B", "TEST_VAR_C", default="nope")
        assert result == "winner"


# ---------------------------------------------------------------------------
# _require_env
# ---------------------------------------------------------------------------


class TestRequireEnv:
    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEST_REQUIRE", "hello")
        assert _require_env("TEST_REQUIRE", "missing") == "hello"

    def test_raises_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TEST_REQUIRE_MISSING", raising=False)
        with pytest.raises(RuntimeError, match="not configured"):
            _require_env("TEST_REQUIRE_MISSING", "not configured")

    def test_raises_when_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEST_REQUIRE_EMPTY", "")
        with pytest.raises(RuntimeError):
            _require_env("TEST_REQUIRE_EMPTY", "empty value")

    def test_raises_when_whitespace_only(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEST_REQUIRE_WS", "   ")
        with pytest.raises(RuntimeError):
            _require_env("TEST_REQUIRE_WS", "whitespace only")


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------


class TestRenderPrompt:
    def test_injects_tenant_profile(self):
        result = render_prompt("Plomberie Test\n- hourly rate: $100")
        assert "Plomberie Test" in result
        assert "hourly rate: $100" in result

    def test_contains_behavioral_instructions(self):
        result = render_prompt("any profile")
        # Should contain key behavioral instructions from the template
        assert "DIAGNOSE" in result
        assert "ASSESS" in result
        assert "PROPOSE" in result
        assert "BOOK" in result
        assert "CONFIRM" in result

    def test_demo_profile_renders_without_error(self):
        result = render_prompt(DEMO_TENANT_PROFILE)
        assert "Plomberie Montferrand" in result

    def test_template_has_placeholder(self):
        assert "{tenant_profile}" in MASTER_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Fallback pricing
# ---------------------------------------------------------------------------


class TestFallbackPricing:
    def test_known_models_have_pricing(self):
        assert "anthropic/claude-sonnet-4.6" in _FALLBACK_PRICING
        assert "anthropic/claude-opus-4.6" in _FALLBACK_PRICING

    def test_pricing_tuples_are_positive(self):
        for model, (inp, out) in _FALLBACK_PRICING.items():
            assert inp > 0, f"{model} input price must be positive"
            assert out > 0, f"{model} output price must be positive"
            assert out > inp, f"{model} output price should exceed input price"

    def test_unknown_model_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        """get_fallback_pricing returns None when the active model has no entry."""
        monkeypatch.setattr(
            "montferrand_agent.agent.get_model_name",
            lambda: "unknown/model-xyz",
        )
        assert get_fallback_pricing() is None

    def test_known_model_returns_tuple(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "montferrand_agent.agent.get_model_name",
            lambda: "anthropic/claude-opus-4.6",
        )
        result = get_fallback_pricing()
        assert result == (5.0, 25.0)

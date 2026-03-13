"""Tests for agent.py — env resolution, prompt rendering, and fallback pricing.

All tests are pure logic — no LLM calls.
"""

import pytest

from montferrand_agent.agent import (
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

    def test_returns_default_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TEST_VAR_NONEXISTENT", raising=False)
        assert _resolve_env("TEST_VAR_NONEXISTENT", default="fallback") == "fallback"

    def test_returns_first_non_empty(self, monkeypatch: pytest.MonkeyPatch):
        """Skips empty strings and whitespace-only, returns first real value."""
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

    @pytest.mark.parametrize(
        "value",
        [None, "", "   "],
        ids=["unset", "empty", "whitespace"],
    )
    def test_raises_when_missing_or_blank(
        self, monkeypatch: pytest.MonkeyPatch, value: str | None
    ):
        if value is None:
            monkeypatch.delenv("TEST_REQUIRE_X", raising=False)
        else:
            monkeypatch.setenv("TEST_REQUIRE_X", value)
        with pytest.raises(RuntimeError):
            _require_env("TEST_REQUIRE_X", "bad value")


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------


class TestRenderPrompt:
    def test_injects_tenant_profile(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MONTFERRAND_TIMEZONE", "America/Montreal")
        result = render_prompt("Plomberie Test\n- hourly rate: $100")
        assert "Plomberie Test" in result
        assert "hourly rate: $100" in result

    def test_template_has_placeholder(self):
        assert "{tenant_profile}" in MASTER_PROMPT_TEMPLATE
        assert "{current_datetime}" in MASTER_PROMPT_TEMPLATE

    def test_profile_with_curly_braces_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """H1 regression: profiles containing { or } must not crash."""
        monkeypatch.setenv("MONTFERRAND_TIMEZONE", "America/Montreal")
        profile = "Hours: {lundi-vendredi} 8h-17h\nNotes: use {{special}} rates"
        result = render_prompt(profile)
        assert "{lundi-vendredi}" in result
        assert "{{special}}" in result

    def test_injects_current_datetime(self, monkeypatch: pytest.MonkeyPatch):
        """The rendered prompt includes date, time, and timezone."""
        monkeypatch.setenv("MONTFERRAND_TIMEZONE", "America/Montreal")
        result = render_prompt("test profile")
        assert "CURRENT DATE AND TIME:" in result
        assert "America/Montreal" in result

    def test_missing_timezone_crashes(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MONTFERRAND_TIMEZONE", raising=False)
        with pytest.raises(RuntimeError, match="MONTFERRAND_TIMEZONE"):
            render_prompt("test profile")


# ---------------------------------------------------------------------------
# Fallback pricing
# ---------------------------------------------------------------------------


class TestFallbackPricing:
    def test_pricing_tuples_are_positive(self):
        for model, (inp, out) in _FALLBACK_PRICING.items():
            assert inp > 0, f"{model} input price must be positive"
            assert out > 0, f"{model} output price must be positive"
            assert out > inp, f"{model} output price should exceed input price"

    def test_unknown_model_returns_none(self, monkeypatch: pytest.MonkeyPatch):
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
        assert get_fallback_pricing() == (5.0, 25.0)

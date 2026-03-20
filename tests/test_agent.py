"""Tests for agent prompts, backend resolution, and structured output setup."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai._agent_graph import ModelRequestNode, UserPromptNode
from pydantic import ValidationError
from pydantic_ai.profiles.openai import OpenAIModelProfile
from zoneinfo import ZoneInfo

from montferrand_agent.agent import (
    AgentDeps,
    DEMO_TENANT_PROFILE,
    MASTER_PROMPT_TEMPLATE,
    _FALLBACK_PRICING,
    _require_env,
    _resolve_env,
    build_model,
    get_fallback_pricing,
    get_agent,
    render_prompt,
    tool_list_schedule,
)
from montferrand_agent.calendar import ListEventsResult
from montferrand_agent.calendar import get_tenant_calendar
from montferrand_agent.llm_backend import (
    DEFAULT_INCEPTION_BASE_URL,
    DEFAULT_OPENROUTER_BASE_URL,
    build_model_profile,
    resolve_backend,
)
from montferrand_agent.models import AgentTurn, Dialog, Report


class TestResolveEnv:
    def test_returns_env_var_when_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEST_VAR_A", "value_a")
        assert _resolve_env("TEST_VAR_A", default="fallback") == "value_a"

    def test_returns_default_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TEST_VAR_NONEXISTENT", raising=False)
        assert _resolve_env("TEST_VAR_NONEXISTENT", default="fallback") == "fallback"

    def test_returns_first_non_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEST_VAR_A", "")
        monkeypatch.setenv("TEST_VAR_B", "  ")
        monkeypatch.setenv("TEST_VAR_C", "winner")
        result = _resolve_env("TEST_VAR_A", "TEST_VAR_B", "TEST_VAR_C", default="nope")
        assert result == "winner"


class TestRequireEnv:
    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEST_REQUIRE", "hello")
        assert _require_env("TEST_REQUIRE", "missing") == "hello"

    @pytest.mark.parametrize(
        "value", [None, "", "   "], ids=["unset", "empty", "whitespace"]
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


class TestRenderPrompt:
    def test_injects_tenant_profile(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MONTFERRAND_TIMEZONE", "America/Montreal")
        result = render_prompt("Plomberie Test\n- hourly rate: $100")
        assert "Plomberie Test" in result
        assert "hourly rate: $100" in result

    def test_template_has_placeholders(self):
        assert "{tenant_profile}" in MASTER_PROMPT_TEMPLATE
        assert "{current_datetime}" in MASTER_PROMPT_TEMPLATE
        assert "{customer_crm_context}" in MASTER_PROMPT_TEMPLATE

    def test_injects_customer_crm_context(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MONTFERRAND_TIMEZONE", "America/Montreal")
        result = render_prompt(
            "Plomberie Test",
            customer_crm_context="- Known customer name: Jonathan Pelletier",
        )
        assert "Known customer name: Jonathan Pelletier" in result

    def test_profile_with_curly_braces_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("MONTFERRAND_TIMEZONE", "America/Montreal")
        profile = "Hours: {lundi-vendredi} 8h-17h\nNotes: use {{special}} rates"
        result = render_prompt(profile)
        assert "{lundi-vendredi}" in result
        assert "{{special}}" in result

    def test_injects_current_datetime(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MONTFERRAND_TIMEZONE", "America/Montreal")
        result = render_prompt("test profile")
        assert "CURRENT DATE AND TIME:" in result
        assert "America/Montreal" in result

    def test_missing_timezone_crashes(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MONTFERRAND_TIMEZONE", raising=False)
        with pytest.raises(RuntimeError, match="MONTFERRAND_TIMEZONE"):
            render_prompt("test profile")

    @pytest.mark.asyncio
    async def test_runtime_model_request_includes_current_datetime(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("MONTFERRAND_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("MONTFERRAND_MODEL", "anthropic/claude-sonnet-4.6")
        monkeypatch.setenv("MONTFERRAND_TIMEZONE", "America/Montreal")
        monkeypatch.setenv("MONTFERRAND_DATA_DIR", str(tmp_path))

        expected_date = datetime.now(ZoneInfo("America/Montreal")).strftime("%Y-%m-%d")
        instructions = render_prompt(DEMO_TENANT_PROFILE)
        agent = get_agent()

        async with agent.iter(
            "bonjour",
            instructions=instructions,
            deps=AgentDeps(calendar=get_tenant_calendar("+15550001111")),
        ) as run:
            node = run.next_node
            if isinstance(node, UserPromptNode):
                node = await run.next(node)

            assert isinstance(node, ModelRequestNode)
            runtime_instructions = node.request.instructions or ""
            assert "CURRENT DATE AND TIME:" in runtime_instructions
            assert expected_date in runtime_instructions
            assert "America/Montreal" in runtime_instructions


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


class TestBackendResolution:
    def test_openrouter_claude_uses_native_strategy(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("MONTFERRAND_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("MONTFERRAND_MODEL", "anthropic/claude-sonnet-4.6")

        backend = resolve_backend()

        assert backend.spec.provider == "openrouter"
        assert backend.spec.base_url == DEFAULT_OPENROUTER_BASE_URL
        assert backend.spec.model_name == "anthropic/claude-sonnet-4.6"
        assert backend.capabilities.structured_output_strategy == "native"

    def test_direct_inception_strips_prefix_and_uses_native(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("MONTFERRAND_PROVIDER", "inception")
        monkeypatch.setenv("INCEPTION_API_KEY", "test-key")
        monkeypatch.setenv("MONTFERRAND_MODEL", "inception/mercury-2")

        backend = resolve_backend()

        assert backend.spec.provider == "inception"
        assert backend.spec.base_url == DEFAULT_INCEPTION_BASE_URL
        assert backend.spec.model_name == "mercury-2"
        assert backend.capabilities.structured_output_strategy == "native"
        assert backend.capabilities.supports_required_tool_choice is False

    def test_older_claude_falls_back_to_tool_strategy(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("MONTFERRAND_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("MONTFERRAND_MODEL", "anthropic/claude-3-haiku-20240307")

        backend = resolve_backend()

        assert backend.capabilities.structured_output_strategy == "tool"
        assert backend.capabilities.supports_required_tool_choice is True

    def test_judge_provider_defaults_to_agent_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("MONTFERRAND_PROVIDER", "inception")
        monkeypatch.setenv("INCEPTION_API_KEY", "test-key")
        monkeypatch.setenv("MONTFERRAND_MODEL", "mercury-2")
        monkeypatch.delenv("MONTFERRAND_JUDGE_PROVIDER", raising=False)
        monkeypatch.delenv("MONTFERRAND_JUDGE_MODEL", raising=False)

        backend = resolve_backend("judge")

        assert backend.spec.provider == "inception"
        assert backend.spec.model_name == "mercury-2"

    def test_judge_provider_can_override_agent_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("MONTFERRAND_PROVIDER", "inception")
        monkeypatch.setenv("INCEPTION_API_KEY", "test-key")
        monkeypatch.setenv("MONTFERRAND_MODEL", "mercury-2")
        monkeypatch.setenv("MONTFERRAND_JUDGE_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "judge-key")
        monkeypatch.setenv("MONTFERRAND_JUDGE_MODEL", "anthropic/claude-sonnet-4.6")

        backend = resolve_backend("judge")

        assert backend.spec.provider == "openrouter"
        assert backend.spec.model_name == "anthropic/claude-sonnet-4.6"


class TestModelProfiles:
    def test_native_strategy_disables_required_tool_choice(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("MONTFERRAND_PROVIDER", "inception")
        monkeypatch.setenv("INCEPTION_API_KEY", "test-key")
        monkeypatch.setenv("MONTFERRAND_MODEL", "mercury-2")

        backend = resolve_backend()
        profile = build_model_profile(backend)

        assert profile.default_structured_output_mode == "native"
        assert profile.openai_supports_tool_choice_required is False

    def test_tool_strategy_keeps_required_tool_choice(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("MONTFERRAND_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("MONTFERRAND_MODEL", "anthropic/claude-3-haiku-20240307")

        backend = resolve_backend()
        profile = build_model_profile(backend)

        assert profile.default_structured_output_mode == "tool"
        assert profile.openai_supports_tool_choice_required is True

    def test_build_model_uses_backend_profile(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MONTFERRAND_PROVIDER", "inception")
        monkeypatch.setenv("INCEPTION_API_KEY", "test-key")
        monkeypatch.setenv("MONTFERRAND_MODEL", "mercury-2")

        model = build_model()
        profile = OpenAIModelProfile.from_profile(model.profile)

        assert model.model_name == "mercury-2"
        assert profile.default_structured_output_mode == "native"
        assert profile.openai_supports_tool_choice_required is False


class TestAgentTurn:
    def test_dialog_turn_converts_to_dialog(self):
        turn = AgentTurn(kind="dialog", message="Bonjour, quel est le probleme?")
        assert turn.to_public_result() == Dialog(
            message="Bonjour, quel est le probleme?"
        )

    def test_report_turn_converts_to_report(self):
        turn = AgentTurn(
            kind="report",
            message="Parfait, c'est reserve.",
            customer_name="Jean Tremblay",
            service_location="123 rue Test",
            issue_description="Drain bouche dans le garage.",
            appointment_window="demain 9h a 12h",
        )
        assert turn.to_public_result() == Report(
            message="Parfait, c'est reserve.",
            customer_name="Jean Tremblay",
            service_location="123 rue Test",
            issue_description="Drain bouche dans le garage.",
            appointment_window="demain 9h a 12h",
        )

    def test_report_turn_requires_all_booking_fields(self):
        with pytest.raises(ValidationError, match="Missing"):
            AgentTurn(
                kind="report",
                message="Parfait, c'est reserve.",
                customer_name="Jean Tremblay",
                service_location=None,
                issue_description="Drain bouche dans le garage.",
                appointment_window="demain 9h a 12h",
            )


class TestToolListSchedule:
    def test_include_past_param_reaches_calendar_backend(self):
        seen: dict[str, object] = {}

        class FakeCalendar:
            def list_events(
                self,
                from_date: str,
                to_date: str,
                include_past: bool = False,
                recent_past_hours: int = 0,
            ) -> ListEventsResult:
                seen["from_date"] = from_date
                seen["to_date"] = to_date
                seen["include_past"] = include_past
                seen["recent_past_hours"] = recent_past_hours
                return ListEventsResult(
                    success=True,
                    message="ok",
                    events=[],
                )

        ctx = cast(
            Any,
            SimpleNamespace(
                deps=cast(Any, AgentDeps(calendar=cast(Any, FakeCalendar())))
            ),
        )
        result = tool_list_schedule(
            ctx,
            "2026-03-01",
            "2026-03-31",
            include_past=True,
            recent_past_hours=4,
        )

        assert result.success is True
        assert seen == {
            "from_date": "2026-03-01",
            "to_date": "2026-03-31",
            "include_past": True,
            "recent_past_hours": 4,
        }

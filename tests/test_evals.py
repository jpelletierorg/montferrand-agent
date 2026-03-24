"""Tests for evals.py — evaluators and report rendering helpers.

All tests are pure logic — no LLM calls.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_evals.evaluators import EvaluationReason

from montferrand_agent.evals import (
    BossEvalResult,
    ConversationResult,
    NoSlowTurns,
    RUBRIC_SMS_STYLE,
    SCENARIOS,
    Scenario,
    ToolUseSmokeResult,
    _RUBRICS,
    _eval_runtime_numbers,
    _display_name,
    _format_transcript_entry,
    _pass_fail,
    main,
    run_scenario,
)
from montferrand_agent.models import Report


# ---------------------------------------------------------------------------
# _pass_fail
# ---------------------------------------------------------------------------


class TestPassFail:
    def test_pass_returns_green(self):
        text = _pass_fail(True)
        assert str(text) == "PASS"
        assert text.style == "bold green"

    def test_fail_returns_red(self):
        text = _pass_fail(False)
        assert str(text) == "FAIL"
        assert text.style == "bold red"


# ---------------------------------------------------------------------------
# _display_name
# ---------------------------------------------------------------------------


class TestDisplayName:
    def test_known_name(self):
        assert _display_name("ConversationConverged") == "Converged"
        assert _display_name("consultative_flow") == "Consult"
        assert _display_name("diagnostic_expertise") == "Diagnostic"
        assert _display_name("explicit_booking_dates") == "Dates"
        assert _display_name("NoSlowTurns") == "Speed"

    def test_unknown_name_returns_itself(self):
        assert _display_name("something_new") == "something_new"


class TestScenarioCoverage:
    def test_self_diagnosed_drain_scenario_is_registered(self):
        scenario = SCENARIOS["self_diagnosed_drain"]
        assert isinstance(scenario, Scenario)
        assert "drain bouché au sous-sol" in scenario.persona


class TestEvalRuntimeNumbers:
    def test_returns_distinct_tenant_and_customer_numbers(self):
        tenant_number, customer_number = _eval_runtime_numbers("scenario-a")

        assert tenant_number.startswith("+1")
        assert customer_number.startswith("+1")
        assert len(tenant_number) == 12
        assert len(customer_number) == 12
        assert tenant_number != customer_number

    def test_different_run_keys_get_different_numbers(self):
        first = _eval_runtime_numbers("scenario-a")
        second = _eval_runtime_numbers("scenario-b")

        assert first != second


class TestTranscriptFormatting:
    def test_format_transcript_entry_marks_message_boundaries(self):
        entry = _format_transcript_entry(
            speaker="AGENT",
            message_index=2,
            message="Bonjour.\n\nDeuxieme paragraphe.",
        )

        assert entry.startswith("[2] AGENT MESSAGE:")
        assert "Deuxieme paragraphe." in entry
        assert entry.endswith("<END MESSAGE>")


class TestJudgeConfiguration:
    def test_only_soft_llm_judges_remain(self):
        names = [name for _rubric, name in _RUBRICS]

        assert names == [
            "diagnostic_expertise",
            "consultative_flow",
            "proactive_proposal",
            "sms_style",
            "natural_tone",
            "preliminary_framing",
            "plain_language",
            "no_assumed_history",
            "greeting_introduction",
            "paragraph_readability",
        ]

    def test_sms_style_rubric_relaxes_question_count(self):
        assert "at most one question" not in RUBRIC_SMS_STYLE
        assert "not feel drowned in questions" in RUBRIC_SMS_STYLE


class TestRunScenarioIsolation:
    @staticmethod
    def _done_report() -> Report:
        return Report(
            message="C'est reserve.",
            customer_name="Test Customer",
            service_location="123 rue Test, Longueuil, J4K 1A1",
            issue_description="Drain bouche.",
            appointment_window="2026-03-24 09:00-12:00",
        )

    @pytest.mark.asyncio
    async def test_run_scenario_uses_isolated_eval_numbers_and_resets_crm(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        scenario = Scenario(persona="Test persona", max_turns=1)
        captured: list[tuple[str, str, str]] = []
        resets: list[str] = []

        monkeypatch.setattr(
            "montferrand_agent.evals.build_customer_agent", lambda _persona: MagicMock()
        )
        monkeypatch.setattr(
            "montferrand_agent.evals.new_conversation_id", lambda: "conv-123"
        )

        async def fake_run_customer_turn(*args, **kwargs):
            del args, kwargs
            return "bonjour", [], None

        async def fake_run_agent_turn(
            conversation_id: str,
            customer_message: str,
            transcript_lines: list[str],
            turns: int,
            tenant_number: str,
            customer_phone: str,
        ):
            del transcript_lines, turns
            captured.append((conversation_id, tenant_number, customer_phone))
            return self._done_report(), 0.1, None

        monkeypatch.setattr(
            "montferrand_agent.evals._run_customer_turn", fake_run_customer_turn
        )
        monkeypatch.setattr(
            "montferrand_agent.evals._run_agent_turn", fake_run_agent_turn
        )
        monkeypatch.setattr(
            "montferrand_agent.evals.reset",
            lambda conversation_id, twilio_number: resets.append(
                f"reset:{conversation_id}:{twilio_number}"
            ),
        )
        monkeypatch.setattr(
            "montferrand_agent.evals.reset_calendar",
            lambda twilio_number: resets.append(f"calendar:{twilio_number}"),
        )
        monkeypatch.setattr(
            "montferrand_agent.evals.reset_tenant_crm",
            lambda twilio_number: resets.append(f"crm:{twilio_number}"),
        )

        result = await run_scenario(scenario)

        expected_tenant, expected_customer = _eval_runtime_numbers("conv-123")
        assert result.report is not None
        assert captured == [("conv-123", expected_tenant, expected_customer)]
        assert f"reset:conv-123:{expected_tenant}" in resets
        assert f"calendar:{expected_tenant}" in resets
        assert f"crm:{expected_tenant}" in resets


# ---------------------------------------------------------------------------
# NoSlowTurns evaluator
# ---------------------------------------------------------------------------


def _make_ctx(
    turn_durations: list[float],
) -> MagicMock:
    """Build a mock EvaluatorContext with the given turn durations."""
    ctx = MagicMock()
    ctx.output = ConversationResult(
        report=None,
        turns=len(turn_durations),
        transcript="",
        turn_durations=turn_durations,
    )
    return ctx


class TestNoSlowTurns:
    def test_all_fast_passes(self):
        evaluator = NoSlowTurns(max_seconds=12.0)
        result = evaluator.evaluate(_make_ctx([2.0, 3.5, 4.1]))
        assert result is True

    def test_one_slow_fails(self):
        evaluator = NoSlowTurns(max_seconds=12.0)
        result = evaluator.evaluate(_make_ctx([2.0, 13.5, 4.1]))
        assert isinstance(result, EvaluationReason)
        assert result.value is False
        assert result.reason is not None
        assert "turn 2" in result.reason

    def test_multiple_slow_lists_all(self):
        evaluator = NoSlowTurns(max_seconds=5.0)
        result = evaluator.evaluate(_make_ctx([6.0, 3.0, 7.0]))
        assert isinstance(result, EvaluationReason)
        assert result.value is False
        assert result.reason is not None
        assert "turn 1" in result.reason
        assert "turn 3" in result.reason

    def test_exact_threshold_passes(self):
        evaluator = NoSlowTurns(max_seconds=12.0)
        result = evaluator.evaluate(_make_ctx([12.0]))
        assert result is True

    def test_empty_durations_passes(self):
        evaluator = NoSlowTurns(max_seconds=12.0)
        result = evaluator.evaluate(_make_ctx([]))
        assert result is True

    def test_custom_threshold(self):
        evaluator = NoSlowTurns(max_seconds=3.0)
        result = evaluator.evaluate(_make_ctx([2.0, 3.5]))
        assert isinstance(result, EvaluationReason)
        assert result.value is False


class TestMainExitBehavior:
    def test_main_exits_non_zero_on_tool_use_smoke_failure(self):
        failing_smoke = ToolUseSmokeResult(
            fixture_name="fixture",
            expected_tool_name="tool_create_service_call",
            expected_location="123 rue Test",
            calls=[],
            output_kind=None,
            output_message=None,
            error=None,
        )
        passing_boss = BossEvalResult(
            scenario_name="boss_french_language_match",
            input_message="quoi?",
            output_message="D'accord.",
            passed=True,
        )

        with (
            patch(
                "montferrand_agent.evals._run_main",
                return_value=(
                    SimpleNamespace(cases=[]),
                    [failing_smoke],
                    [passing_boss],
                ),
            ),
            patch("montferrand_agent.evals.print_report"),
            patch("montferrand_agent.evals.print_tool_use_report"),
            patch("montferrand_agent.evals.print_boss_eval_report"),
        ):
            try:
                main(model_name=None)
            except SystemExit as exc:
                assert exc.code == 1
            else:
                raise AssertionError("Expected SystemExit(1)")

    def test_main_exits_non_zero_on_boss_eval_failure(self):
        passing_smoke = ToolUseSmokeResult(
            fixture_name="fixture",
            expected_tool_name="tool_check_availability",
            expected_location="",
            calls=[],
            output_kind=None,
            output_message=None,
            error=None,
        )
        failing_boss = BossEvalResult(
            scenario_name="boss_french_language_match",
            input_message="quoi?",
            output_message="Hello",
            passed=False,
            error="Boss reply switched to English.",
        )

        with (
            patch(
                "montferrand_agent.evals._run_main",
                return_value=(
                    SimpleNamespace(cases=[]),
                    [passing_smoke],
                    [failing_boss],
                ),
            ),
            patch("montferrand_agent.evals.print_report"),
            patch("montferrand_agent.evals.print_tool_use_report"),
            patch("montferrand_agent.evals.print_boss_eval_report"),
        ):
            try:
                main(model_name=None)
            except SystemExit as exc:
                assert exc.code == 1
            else:
                raise AssertionError("Expected SystemExit(1)")

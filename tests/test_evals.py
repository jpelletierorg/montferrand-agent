"""Tests for evals.py — evaluators and report rendering helpers.

All tests are pure logic — no LLM calls.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_evals.evaluators import EvaluationReason

from montferrand_agent.evals import (
    BossEvalResult,
    ConversationResult,
    NoSlowTurns,
    Scenario,
    ToolUseSmokeResult,
    _display_name,
    _pass_fail,
    main,
)


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
        assert _display_name("diagnostic_expertise") == "Diagnostic"
        assert _display_name("explicit_booking_dates") == "Dates"
        assert _display_name("NoSlowTurns") == "Speed"

    def test_unknown_name_returns_itself(self):
        assert _display_name("something_new") == "something_new"


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

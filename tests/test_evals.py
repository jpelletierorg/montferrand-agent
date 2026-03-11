"""Tests for evals.py — evaluators and report rendering helpers.

All tests are pure logic — no LLM calls.
"""

from unittest.mock import MagicMock

from pydantic_evals.evaluators import EvaluationReason

from montferrand_agent.evals import (
    ConversationResult,
    NoSlowTurns,
    Scenario,
    _display_name,
    _pass_fail,
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
        assert _display_name("ConversationConverged") == "Conv."
        assert _display_name("diagnostic_expertise") == "Diag."
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

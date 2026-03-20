"""Tests for raw latency benchmark helpers."""

from montferrand_agent.latency import LatencyReport, LatencySample


def test_latency_report_aggregates_successful_samples_only():
    report = LatencyReport(
        provider="openrouter",
        model_name="anthropic/claude-sonnet-4.6",
        prompt="What is the meaning of life? Reply in one short sentence.",
        instruction_chars=3200,
        wall_seconds=5.5,
        samples=[
            LatencySample(
                index=1, elapsed_seconds=1.0, success=True, response_text="42."
            ),
            LatencySample(
                index=2,
                elapsed_seconds=3.0,
                success=True,
                response_text="To live with purpose.",
            ),
            LatencySample(index=3, elapsed_seconds=9.0, success=False, error="boom"),
        ],
    )

    assert report.success_count == 2
    assert report.failure_count == 1
    assert report.average_latency_seconds == 2.0
    assert report.min_latency_seconds == 1.0
    assert report.max_latency_seconds == 3.0


def test_latency_report_handles_all_failures():
    report = LatencyReport(
        provider="openrouter",
        model_name="anthropic/claude-opus-4.6",
        prompt="What is the meaning of life? Reply in one short sentence.",
        instruction_chars=3200,
        wall_seconds=4.0,
        samples=[
            LatencySample(index=1, elapsed_seconds=2.0, success=False, error="x"),
            LatencySample(index=2, elapsed_seconds=2.5, success=False, error="y"),
        ],
    )

    assert report.success_count == 0
    assert report.failure_count == 2
    assert report.average_latency_seconds is None
    assert report.min_latency_seconds is None
    assert report.max_latency_seconds is None

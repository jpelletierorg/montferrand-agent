"""Tests for CLI tenant onboarding flows."""

import os
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from montferrand_agent.calendar import get_tenant_calendar
from montferrand_agent.cli import _build_cli_banner_text, _format_trace_event, app
from montferrand_agent.conversation import ConversationTraceEvent
from montferrand_agent.crm import tenant_crm_db_path
from montferrand_agent.ops import (
    ReadinessCheck,
    mark_outbound_accepted,
    mark_outbound_attempted,
    mark_processing_started,
    mark_processing_succeeded,
    record_inbound_received,
)
from montferrand_agent.tenant import load_tenant_profile

from .conftest import TEST_PROFILE, TWILIO_NUMBER


def test_cli_banner_includes_backend_details():
    text = _build_cli_banner_text("boss", "mercury-2", "inception", "native")

    assert "Role: [dim]boss[/dim]" in text
    assert "Provider: [dim]inception[/dim]" in text
    assert "Model: [dim]mercury-2[/dim]" in text
    assert "Structured output: [dim]native[/dim]" in text


def test_trace_event_formatting():
    event = ConversationTraceEvent(
        kind="request_finished",
        summary='finish=tool_calls; tools=tool_list_events; text="Bonjour"',
        at_seconds=12.34,
        request_index=2,
        elapsed_seconds=4.56,
    )

    text = _format_trace_event(event)

    assert text == (
        "t+12.3s request #2 finished in 4.6s - "
        'finish=tool_calls; tools=tool_list_events; text="Bonjour"'
    )


def test_cli_trace_flag_wires_into_loop():
    seen: list[tuple[str, bool]] = []

    async def fake_loop(*, agent_role: str = "customer", trace: bool = False):
        seen.append((agent_role, trace))

    runner = CliRunner()
    with patch("montferrand_agent.cli._cli_loop", new=fake_loop):
        result = runner.invoke(app, ["cli", "--trace"])

    assert result.exit_code == 0
    assert seen == [("customer", True)]


def test_cli_agent_flag_wires_boss_into_loop():
    seen: list[tuple[str, bool]] = []

    async def fake_loop(*, agent_role: str = "customer", trace: bool = False):
        seen.append((agent_role, trace))

    runner = CliRunner()
    with patch("montferrand_agent.cli._cli_loop", new=fake_loop):
        result = runner.invoke(app, ["cli", "--agent", "boss"])

    assert result.exit_code == 0
    assert seen == [("boss", False)]


def test_cli_model_flag_overrides_env(monkeypatch):
    seen: list[tuple[str, str | None]] = []
    monkeypatch.setenv("MONTFERRAND_MODEL", "anthropic/claude-sonnet-4.6")

    async def fake_loop(*, agent_role: str = "customer", trace: bool = False):
        import os

        del agent_role
        del trace
        seen.append(("customer", os.environ.get("MONTFERRAND_MODEL")))

    runner = CliRunner()
    with patch("montferrand_agent.cli._cli_loop", new=fake_loop):
        result = runner.invoke(app, ["cli", "--model", "inception/mercury-2"])

    assert result.exit_code == 0
    assert seen == [("customer", "inception/mercury-2")]
    assert os.environ.get("MONTFERRAND_MODEL") == "anthropic/claude-sonnet-4.6"


def test_latency_command_wires_into_async_runner():
    seen: list[tuple[str | None, str | None, int]] = []

    async def fake_latency_command(
        *, model_name: str | None, provider: str | None, samples: int
    ):
        seen.append((model_name, provider, samples))

    runner = CliRunner()
    with patch("montferrand_agent.cli._latency_command", new=fake_latency_command):
        result = runner.invoke(
            app,
            [
                "latency",
                "--model",
                "anthropic/claude-opus-4.6",
                "--provider",
                "openrouter",
                "--samples",
                "12",
            ],
        )

    assert result.exit_code == 0
    assert seen == [("anthropic/claude-opus-4.6", "openrouter", 12)]


def test_onboard_creates_calendar_directory(
    isolated_data_dir: Path, tmp_path: Path, fake_dbmate
):
    prompt_file = tmp_path / "profile.txt"
    prompt_file.write_text(TEST_PROFILE, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "onboard",
            "--twilio-number",
            TWILIO_NUMBER,
            "--prompt-file",
            str(prompt_file),
            "--local",
        ],
    )

    assert result.exit_code == 0
    assert load_tenant_profile(TWILIO_NUMBER) == TEST_PROFILE
    assert get_tenant_calendar(TWILIO_NUMBER).directory.exists()
    assert tenant_crm_db_path(TWILIO_NUMBER).exists()


def test_onboard_rejects_existing_tenant(
    isolated_data_dir: Path, tmp_path: Path, fake_dbmate
):
    prompt_file = tmp_path / "profile.txt"
    prompt_file.write_text(TEST_PROFILE, encoding="utf-8")

    runner = CliRunner()
    first = runner.invoke(
        app,
        [
            "onboard",
            "--twilio-number",
            TWILIO_NUMBER,
            "--prompt-file",
            str(prompt_file),
            "--local",
        ],
    )
    second = runner.invoke(
        app,
        [
            "onboard",
            "--twilio-number",
            TWILIO_NUMBER,
            "--prompt-file",
            str(prompt_file),
            "--local",
        ],
    )

    assert first.exit_code == 0
    assert second.exit_code == 1
    assert "Tenant already exists" in second.output


def test_serve_runs_crm_migrations_before_starting_server():
    runner = CliRunner()

    with (
        patch(
            "montferrand_agent.cli.migrate_all_tenant_crm",
            return_value=[Path("/tmp/a")],
        ),
        patch("uvicorn.run") as mock_run,
    ):
        result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_reset_command_mentions_calendar():
    runner = CliRunner()

    with patch("montferrand_agent.cli.reset_tenant", return_value=2):
        result = runner.invoke(
            app,
            ["reset", "--twilio-number", TWILIO_NUMBER, "--local"],
        )

    assert result.exit_code == 0
    assert "Deleted 2 conversation(s) and reset the calendar" in result.output


def test_ops_doctor_command_uses_readiness_checks():
    runner = CliRunner()

    with patch(
        "montferrand_agent.cli.run_readiness_checks",
        return_value=[ReadinessCheck("ops_db", True, "ok")],
    ):
        result = runner.invoke(app, ["ops", "doctor"])

    assert result.exit_code == 0
    assert "Ops Doctor" in result.output
    assert "ops_db" in result.output


def test_ops_incident_renders_timeline(isolated_data_dir: Path):
    record_inbound_received(
        message_id="SMCLI1",
        message_sid="SMCLI1",
        conversation_id="conv-1",
        twilio_number=TWILIO_NUMBER,
        from_number="+14380000000",
        body="Besoin d'un plombier",
    )
    mark_processing_started("SMCLI1", is_boss=False)
    mark_processing_succeeded("SMCLI1", "Bonjour!")
    mark_outbound_attempted("SMCLI1", "Bonjour!")
    mark_outbound_accepted("SMCLI1", "SMOUT1")

    runner = CliRunner()
    result = runner.invoke(app, ["ops", "incident", "--message-sid", "SMCLI1"])

    assert result.exit_code == 0
    assert "Timeline" in result.output
    assert "outbound_accepted" in result.output

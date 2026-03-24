"""CLI entry point for the Montferrand booking agent.

Provides subcommands:

    uv run montferrand cli                — interactive conversation loop
    uv run montferrand crm provision      — provision tenant CRM database
    uv run montferrand latency            — benchmark raw model latency
    uv run montferrand evals              — run the eval suite
    uv run montferrand serve              — start the webhook server
    uv run montferrand onboard            — register a new tenant
    uv run montferrand tenant edit        — edit a tenant's prompt
    uv run montferrand tenant list        — list configured tenants
    uv run montferrand calendar          — show booked events for a tenant
    uv run montferrand reset              — wipe conversation data and reset calendar
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, NoReturn, TypeAlias

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from montferrand_agent.agent import (
    DEMO_TENANT_PROFILE,
    get_model_name,
    get_provider_name,
    get_structured_output_strategy,
)
from montferrand_agent.conversation import (
    ConversationTraceEvent,
    ConversationCost,
    ConversationError,
    get_cost,
    new_conversation_id,
    process_message,
    reset,
    reset_tenant,
)
from montferrand_agent.crm import (
    ensure_existing_tenant_crm,
    ensure_tenant_crm,
    TenantCrmError,
    migrate_all_tenant_crm,
    provision_all_tenant_crm,
    provision_tenant_crm,
    verify_all_tenant_crm,
)
from montferrand_agent.latency import LatencyReport, run_latency_benchmark
from montferrand_agent.models import Report
from montferrand_agent.next_work_item import maybe_handle_next_work_item_command
from montferrand_agent.ops import (
    find_messages,
    get_message_timeline,
    run_readiness_checks,
)
from montferrand_agent.tenant import (
    TenantConfig,
    TenantNotFoundError,
    list_tenants,
    load_tenant_config,
    load_tenant_profile,
    save_tenant_config,
    tenant_exists,
)

app = typer.Typer(
    name="montferrand",
    help="Montferrand booking agent CLI.",
    add_completion=False,
)
tenant_app = typer.Typer(
    name="tenant",
    help="Manage tenant configurations.",
    add_completion=False,
)
crm_app = typer.Typer(
    name="crm",
    help="Manage tenant CRM databases.",
    add_completion=False,
)
ops_app = typer.Typer(
    name="ops",
    help="Operations and incident tooling.",
    add_completion=False,
)
app.add_typer(tenant_app, name="tenant")
app.add_typer(crm_app, name="crm")
app.add_typer(ops_app, name="ops")
console = Console()
CliAgentRole: TypeAlias = Literal["customer", "boss"]
_REMOTE_ADMIN_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=120.0,
    write=30.0,
    pool=10.0,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _print_error(message: str) -> None:
    """Render an error message (non-fatal)."""
    console.print(f"[red]{message}[/red]")


def _fatal(message: str) -> NoReturn:
    """Render an error message and exit with code 1."""
    _print_error(message)
    raise typer.Exit(1)


def _format_incident_preview(text: str | None, limit: int = 70) -> str:
    """Return a compact one-line preview for incident output."""

    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _render_incident_timeline(
    message_sid: str | None = None,
    message_id: str | None = None,
) -> None:
    """Render one incident timeline from the ops store."""

    timeline = get_message_timeline(message_sid=message_sid, message_id=message_id)
    if timeline is None:
        lookup = message_sid or message_id or "<unknown>"
        _fatal(f"No incident record found for {lookup}")

    message, events = timeline
    summary = Table(show_header=False, box=None)
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Inbound SID", message.message_sid or "-")
    summary.add_row("Message ID", message.message_id)
    summary.add_row("Conversation", message.conversation_id)
    summary.add_row("Tenant", message.twilio_number)
    summary.add_row("Customer", message.from_number)
    summary.add_row("Boss Flow", "yes" if message.is_boss else "no")
    summary.add_row("Last Stage", message.last_stage)
    summary.add_row("Updated", message.updated_at)
    if message.outbound_message_sid:
        summary.add_row("Outbound SID", message.outbound_message_sid)
    if message.error_text:
        summary.add_row(
            "Error", _format_incident_preview(message.error_text, limit=120)
        )
    if message.reply_body:
        summary.add_row(
            "Reply", _format_incident_preview(message.reply_body, limit=120)
        )

    console.print(summary)
    console.print()

    event_table = Table(title="Timeline", show_lines=False)
    event_table.add_column("When", style="dim")
    event_table.add_column("Event", style="bold")
    event_table.add_column("Summary")
    for event in events:
        event_table.add_row(event.created_at, event.event_kind, event.summary)
    console.print(event_table)


@contextlib.contextmanager
def _temporary_env(**overrides: str | None):
    """Temporarily override environment variables for one CLI command."""

    original: dict[str, str | None] = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                continue
            os.environ[name] = value
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _require_admin_token() -> str:
    """Return the admin token or exit with an error."""
    token = os.environ.get("MONTFERRAND_ADMIN_TOKEN", "").strip()
    if not token:
        _fatal("MONTFERRAND_ADMIN_TOKEN is not set.")
    return token


def _get_editor() -> str:
    """Return the user's preferred editor."""
    return os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))


def _edit_text_in_editor(initial_text: str, suffix: str = ".txt") -> str | None:
    """Open *initial_text* in the user's editor and return the result.

    Returns ``None`` if the user saved an empty file or the editor exited
    with a non-zero status.
    """
    editor = _get_editor()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as f:
        f.write(initial_text)
        tmp_path = f.name

    try:
        result = subprocess.run([editor, tmp_path], check=False)
        if result.returncode != 0:
            return None
        edited = Path(tmp_path).read_text(encoding="utf-8")
        return edited if edited.strip() else None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _resolve_host(host: str | None, *, local: bool = False) -> str | None:
    """Return the remote host from the CLI flag or ``MONTFERRAND_HOST``.

    If *local* is True, always return None (force local writes).
    """
    if local:
        return None
    if host:
        return host
    return os.environ.get("MONTFERRAND_HOST", "").strip() or None


def _remote_admin_url(host: str, path: str) -> str:
    return f"https://{host}{path}"


def _request_remote_admin(
    method: Literal["GET", "POST", "DELETE"],
    host: str,
    path: str,
    *,
    json_body: dict[str, str | list[str]] | None = None,
) -> httpx.Response:
    token = _require_admin_token()
    headers = {"Authorization": f"Bearer {token}"}
    url = _remote_admin_url(host, path)

    try:
        if method == "GET":
            return httpx.get(url, headers=headers, timeout=_REMOTE_ADMIN_TIMEOUT)
        if method == "DELETE":
            return httpx.delete(url, headers=headers, timeout=_REMOTE_ADMIN_TIMEOUT)
        return httpx.post(
            url,
            json=json_body,
            headers=headers,
            timeout=_REMOTE_ADMIN_TIMEOUT,
        )
    except httpx.TimeoutException:
        _fatal(
            f"Timed out waiting for {host}. The server may still be starting after "
            f"a deploy. Try again in a minute or check {_remote_admin_url(host, '/health')}."
        )
    except httpx.RequestError as exc:
        _fatal(f"Could not reach {host}: {exc}")


def _push_to_remote(
    host: str,
    twilio_number: str,
    profile: str,
    boss_numbers: list[str] | None = None,
) -> None:
    """POST a tenant config to a remote Montferrand server."""
    payload: dict[str, str | list[str]] = {
        "twilio_number": twilio_number,
        "tenant_profile": profile,
    }
    if boss_numbers:
        payload["boss_numbers"] = boss_numbers

    response = _request_remote_admin(
        "POST",
        host,
        "/admin/tenants",
        json_body=payload,
    )
    if response.status_code in {200, 201}:
        console.print(f"[green]Tenant pushed to {host}[/green]")
    else:
        _fatal(f"Remote error {response.status_code}: {response.text}")


def _prepare_local_tenant_resources(twilio_number: str, *, create: bool) -> Path:
    from montferrand_agent.calendar import ensure_tenant_calendar

    ensure_tenant_calendar(twilio_number)
    try:
        if create:
            return provision_tenant_crm(twilio_number)
        return ensure_tenant_crm(twilio_number)
    except TenantCrmError as exc:
        _fatal(str(exc))


# ---------------------------------------------------------------------------
# cli subcommand
# ---------------------------------------------------------------------------

HELP_TEXT = """\
Commands:
  !attach <path> [message]  — attach an image with an optional message
  !reset                    — start a new conversation
  !quit                     — exit\
"""

_CONVERSATION_OVER_MSG = (
    "[dim]Conversation terminee. Tapez !reset pour recommencer.[/dim]"
)
_DIALOG_IN_PROGRESS_MSG = "[dim][Dialog — conversation en cours][/dim]"
_CLI_CUSTOMER_PHONE = "+15550000001"


def _format_token_usage(cost: ConversationCost) -> str:
    """Return a compact token usage string."""
    return f"{cost.usage.input_tokens} in / {cost.usage.output_tokens} out"


def _parse_attach_command(stripped: str) -> tuple[str, list[Path]]:
    """Parse a regular message or `!attach` command.

    Raises:
        ValueError: If the command is malformed or the file is missing.
    """
    if not stripped.lower().startswith("!attach "):
        return stripped, []

    parts = stripped[len("!attach ") :].strip()
    if not parts:
        raise ValueError("Usage: !attach <chemin> [message]")

    tokens = parts.split(maxsplit=1)
    image_path = Path(tokens[0]).expanduser()
    if not image_path.exists():
        raise ValueError(f"Fichier introuvable: {image_path}")

    text = tokens[1] if len(tokens) > 1 else ""
    return text, [image_path]


def _print_agent_message(message: str) -> None:
    """Render the agent's SMS reply."""
    agent_text = Text()
    agent_text.append("Montferrand > ", style="bold green")
    agent_text.append(message)
    console.print(agent_text)


def _print_report(report: Report) -> None:
    """Render the final booking report."""
    console.print(
        Panel(
            f"[bold]Client:[/bold]      {report.customer_name}\n"
            f"[bold]Adresse:[/bold]     {report.service_location}\n"
            f"[bold]Probleme:[/bold]    {report.issue_description}\n"
            f"[bold]Rendez-vous:[/bold] {report.appointment_window}",
            title="RAPPORT DE SERVICE",
            border_style="green",
        )
    )


def _print_cost(cost: ConversationCost) -> None:
    """Display conversation cost estimate."""
    if cost.usage.total_tokens == 0:
        return

    tokens = _format_token_usage(cost)
    if cost.cost_available:
        console.print(f"[dim]Cout estime: ${cost.total_usd:.4f} USD ({tokens})[/dim]")
    else:
        console.print(
            f"[dim]Tokens utilises: {tokens} (cout non disponible pour ce modele)[/dim]"
        )


def _build_cli_banner_text(
    agent_role: CliAgentRole,
    model_name: str,
    provider_name: str,
    structured_output_strategy: str,
) -> str:
    """Build the interactive CLI banner text."""

    title = "Agent SMS" if agent_role == "customer" else "Agent Boss"

    return (
        f"[bold]Plomberie Montferrand[/bold] — {title} (demo)\n"
        f"Role: [dim]{agent_role}[/dim]\n"
        f"Provider: [dim]{provider_name}[/dim]\n"
        f"Model: [dim]{model_name}[/dim]\n"
        f"Structured output: [dim]{structured_output_strategy}[/dim]\n\n" + HELP_TEXT
    )


def _print_backend_notice(provider_name: str, structured_output_strategy: str) -> None:
    """Display a small notice when the active backend details change."""

    console.print(
        f"[dim]Backend actif: {provider_name} / {structured_output_strategy}[/dim]"
    )


def _format_trace_event(event: ConversationTraceEvent) -> str:
    """Format a conversation trace event for CLI output."""

    prefix = f"t+{event.at_seconds:.1f}s"

    if event.kind == "request_started":
        return f"{prefix} request #{event.request_index} started - {event.summary}"

    if event.kind == "request_finished":
        return (
            f"{prefix} request #{event.request_index} finished in "
            f"{event.elapsed_seconds:.1f}s - {event.summary}"
        )

    if event.kind == "tool_called":
        return f"{prefix} tool call - {event.summary}"

    if event.kind == "tool_result":
        return (
            f"{prefix} tool result - {event.tool_name} in "
            f"{event.elapsed_seconds:.1f}s - {event.summary}"
        )

    if event.kind == "tool_retry":
        return (
            f"{prefix} tool retry - {event.tool_name or 'result'} in "
            f"{event.elapsed_seconds:.1f}s - {event.summary}"
        )

    if event.kind == "warning":
        return f"{prefix} warning - {event.summary}"

    return f"{prefix} turn finished in {event.elapsed_seconds:.1f}s - {event.summary}"


def _print_trace_event(event: ConversationTraceEvent) -> None:
    """Render a trace event in a subdued CLI style."""

    console.print(f"[dim]trace[/dim] {_format_trace_event(event)}")


def _format_latency_seconds(seconds: float | None) -> str:
    """Render a latency duration with one decimal place."""

    if seconds is None:
        return "-"
    return f"{seconds:.2f}s"


def _preview_latency_error(error: str | None, limit: int = 80) -> str:
    """Return a compact single-line error preview for the latency table."""

    if not error:
        return ""
    compact = " ".join(error.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _print_latency_report(report: LatencyReport) -> None:
    """Render the raw model latency benchmark report."""

    console.print()
    console.print("[bold]Latency Benchmark[/bold]")
    console.print(
        "[dim]Raw model benchmark: plain text only, no Montferrand prompt, no tools.[/dim]"
    )
    console.print(f"Provider: [bold]{report.provider}[/bold]")
    console.print(f"Model: [bold]{report.model_name}[/bold]")
    console.print(f"Prompt: [dim]{report.prompt}[/dim]")
    console.print(f"Instruction context: [bold]{report.instruction_chars} chars[/bold]")
    console.print(f"Parallel requests: [bold]{len(report.samples)}[/bold]")
    console.print()

    table = Table(show_header=True)
    table.add_column("Request", justify="right")
    table.add_column("Status")
    table.add_column("Latency", justify="right")
    table.add_column("Details")

    for sample in report.samples:
        table.add_row(
            str(sample.index),
            "ok" if sample.success else "error",
            _format_latency_seconds(sample.elapsed_seconds),
            "" if sample.success else _preview_latency_error(sample.error),
        )

    console.print(table)
    console.print()
    console.print(
        f"Average latency: [bold]{_format_latency_seconds(report.average_latency_seconds)}[/bold]"
    )
    console.print(
        "Min / Max: "
        f"[bold]{_format_latency_seconds(report.min_latency_seconds)}[/bold] / "
        f"[bold]{_format_latency_seconds(report.max_latency_seconds)}[/bold]"
    )
    console.print(
        f"Wall clock: [bold]{_format_latency_seconds(report.wall_seconds)}[/bold]"
    )
    console.print(
        f"Successes: [bold]{report.success_count}[/bold] / {len(report.samples)}"
    )

    console.print()
    console.print("[bold]Responses[/bold]")
    for sample in report.samples:
        if sample.success:
            response = sample.response_text or ""
            console.print(f"{sample.index}. {response}")
        else:
            console.print(
                f"{sample.index}. [red]ERROR:[/red] {_preview_latency_error(sample.error, limit=200)}"
            )


async def _latency_command(
    *,
    model_name: str | None,
    provider: str | None,
    samples: int,
) -> None:
    """Run the raw model latency benchmark and print the report."""

    report = await run_latency_benchmark(
        model_name=model_name,
        provider=provider,  # type: ignore[arg-type]
        samples=samples,
    )
    _print_latency_report(report)

    if report.success_count == 0:
        raise typer.Exit(1)


def _end_conversation(conversation_id: str) -> None:
    """Print the current conversation cost summary."""
    _print_cost(get_cost(conversation_id))


def _resolve_cli_tenant() -> tuple[str, str]:
    """Load the demo tenant's profile and number for the interactive CLI.

    Reads ``MONTFERRAND_DEMO_TENANT`` and loads the corresponding tenant
    profile from disk.  Crashes if the env var is not set or the tenant
    is not found — there is no silent fallback.

    Returns:
        (twilio_number, tenant_profile) tuple.
    """
    demo_number = os.environ.get("MONTFERRAND_DEMO_TENANT", "").strip()
    if not demo_number:
        _fatal(
            "MONTFERRAND_DEMO_TENANT is not set. "
            "Set it to a Twilio number with a configured tenant profile."
        )
    try:
        profile = load_tenant_profile(demo_number)
    except TenantNotFoundError:
        _fatal(
            f"Demo tenant {demo_number} not found. "
            f"Run 'montferrand onboard' to create it first."
        )
    return demo_number, profile


@app.command()
def cli(
    agent: CliAgentRole = typer.Option(
        "customer",
        "--agent",
        help="Which interactive agent to run: customer or boss.",
    ),
    trace: bool = typer.Option(
        False,
        "--trace",
        help="Show live model and tool activity while a turn is running.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Override the model name for this CLI session.",
    ),
) -> None:
    """Interactive conversation with the Montferrand booking agent."""
    with _temporary_env(MONTFERRAND_MODEL=model):
        asyncio.run(_cli_loop(agent_role=agent, trace=trace))


@app.command()
def latency(
    model: str | None = typer.Option(
        None,
        "--model",
        help="Override the model name for this benchmark run.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Override the backend provider: openrouter or inception.",
    ),
    samples: int = typer.Option(
        10,
        "--samples",
        min=1,
        help="Number of parallel requests to launch.",
    ),
) -> None:
    """Measure raw model latency with parallel plain-text requests."""

    asyncio.run(_latency_command(model_name=model, provider=provider, samples=samples))


async def _cli_loop(
    *, agent_role: CliAgentRole = "customer", trace: bool = False
) -> None:
    """Async interactive conversation loop."""
    try:
        model_name = get_model_name()
        provider_name = get_provider_name()
        structured_output_strategy = get_structured_output_strategy()
    except Exception as exc:
        _print_error(f"Erreur de configuration: {exc}")
        return

    twilio_number, tenant_profile = _resolve_cli_tenant()

    console.print(
        Panel(
            _build_cli_banner_text(
                agent_role,
                model_name,
                provider_name,
                structured_output_strategy,
            ),
            border_style="blue",
        )
    )

    conversation_id = new_conversation_id()
    conversation_over = False

    while True:
        # Prompt ----------------------------------------------------------
        try:
            user_input = console.input("[bold cyan]Vous >[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            _end_conversation(conversation_id)
            break

        stripped = user_input.strip()

        # Commands --------------------------------------------------------
        if stripped.lower() in {"!quit", "!q"}:
            _end_conversation(conversation_id)
            break

        if stripped.lower() == "!reset":
            _end_conversation(conversation_id)
            reset(conversation_id, twilio_number)
            conversation_id = new_conversation_id()
            conversation_over = False
            console.print("[dim]Conversation reinitialised.[/dim]\n")
            continue

        if conversation_over:
            console.print(_CONVERSATION_OVER_MSG)
            continue

        # Parse !attach ---------------------------------------------------
        try:
            text, images = _parse_attach_command(stripped)
        except ValueError as exc:
            _print_error(str(exc))
            continue

        if images:
            console.print(f"[dim]Image jointe: {images[0].name}[/dim]")

        if not text and not images:
            continue

        if agent_role == "boss" and text and not images:
            shortcut_reply = await maybe_handle_next_work_item_command(
                twilio_number,
                text,
                is_boss=True,
            )
            if shortcut_reply is not None:
                _print_agent_message(shortcut_reply)
                console.print()
                continue

        # Process turn ----------------------------------------------------
        try:
            result = await process_message(
                conversation_id,
                text,
                images or None,
                tenant_profile=tenant_profile,
                twilio_number=twilio_number,
                is_boss=agent_role == "boss",
                customer_phone=None if agent_role == "boss" else _CLI_CUSTOMER_PHONE,
                trace_observer=_print_trace_event if trace else None,
            )
        except ConversationError as exc:
            _print_error(f"Erreur: {exc}")
            continue

        # Display agent reply ---------------------------------------------
        _print_agent_message(result.message)

        # Display report if conversation is complete ----------------------
        if isinstance(result, Report):
            conversation_over = True
            _print_report(result)
            _end_conversation(conversation_id)
            console.print(_CONVERSATION_OVER_MSG)
        else:
            console.print(_DIALOG_IN_PROGRESS_MSG)

        current_provider_name = get_provider_name()
        current_strategy = get_structured_output_strategy()
        if (
            current_provider_name != provider_name
            or current_strategy != structured_output_strategy
        ):
            provider_name = current_provider_name
            structured_output_strategy = current_strategy
            _print_backend_notice(provider_name, structured_output_strategy)

        console.print()


# ---------------------------------------------------------------------------
# serve subcommand
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8080, help="Port number"),
) -> None:
    """Start the Montferrand webhook server."""
    import uvicorn

    try:
        migrated, provisioned = ensure_existing_tenant_crm()
    except TenantCrmError as exc:
        _fatal(str(exc))

    if migrated:
        console.print(f"[green]CRM migrated for {len(migrated)} tenant(s).[/green]")
    if provisioned:
        console.print(
            "[green]CRM provisioned for tenant(s) missing a database:[/green] "
            + ", ".join(provisioned)
        )

    uvicorn.run(
        "montferrand_agent.server:app",
        host=host,
        port=port,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# onboard subcommand
# ---------------------------------------------------------------------------


def _build_example_toml(twilio_number: str) -> str:
    """Build an example TOML config for a new tenant."""
    import tomli_w

    data: dict = {
        "phone": twilio_number,
        "boss_numbers": [],
        "profile": {"text": DEMO_TENANT_PROFILE.rstrip()},
    }
    return tomli_w.dumps(data)


@app.command()
def onboard(
    twilio_number: str = typer.Option(
        ...,
        "--twilio-number",
        "-n",
        prompt="Twilio phone number (E.164)",
    ),
    prompt_file: Path | None = typer.Option(
        None,
        "--prompt-file",
        "-f",
        help="Read tenant profile from file instead of editor",
    ),
    boss_numbers: str | None = typer.Option(
        None,
        "--boss-numbers",
        "-b",
        help="Comma-separated list of boss phone numbers (E.164)",
    ),
    host: str | None = typer.Option(
        None,
        "--host",
        help="Push to a remote server (falls back to MONTFERRAND_HOST)",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Write locally even if MONTFERRAND_HOST is set",
    ),
) -> None:
    """Register a new tenant with a company profile."""
    remote = _resolve_host(host, local=local)

    if not remote and tenant_exists(twilio_number):
        _fatal(
            f"Tenant already exists for {twilio_number}. Use `montferrand tenant edit` instead."
        )

    parsed_boss: list[str] = []
    if boss_numbers:
        parsed_boss = [n.strip() for n in boss_numbers.split(",") if n.strip()]

    # Load profile from file or open editor with TOML template
    if prompt_file:
        if not prompt_file.exists():
            _fatal(f"File not found: {prompt_file}")
        profile = prompt_file.read_text(encoding="utf-8")
    else:
        example_toml = _build_example_toml(twilio_number)
        edited = _edit_text_in_editor(example_toml, suffix=".toml")
        if edited is None:
            _fatal("Aborted — empty or unchanged config.")

        # Parse the TOML to extract profile and boss_numbers
        import tomllib

        try:
            data = tomllib.loads(edited)
        except Exception as exc:
            _fatal(f"Invalid TOML: {exc}")
        profile = data.get("profile", {}).get("text", "").strip()
        if not profile:
            _fatal("Aborted — empty profile text.")
        # Boss numbers from TOML override the --boss-numbers flag
        toml_boss = data.get("boss_numbers", [])
        if toml_boss:
            parsed_boss = toml_boss

    # Save locally or push to remote
    if remote:
        _push_to_remote(remote, twilio_number, profile, parsed_boss)
    else:
        config = TenantConfig(
            phone=twilio_number,
            profile=profile,
            boss_numbers=parsed_boss,
        )
        db_path = _prepare_local_tenant_resources(twilio_number, create=True)
        path = save_tenant_config(config)
        console.print(f"[green]Tenant saved:[/green] {path}")
        console.print(f"[green]CRM ready:[/green] {db_path}")


# ---------------------------------------------------------------------------
# tenant subcommand group
# ---------------------------------------------------------------------------


@tenant_app.command("edit")
def tenant_edit(
    twilio_number: str = typer.Option(
        ...,
        "--twilio-number",
        "-n",
        help="Tenant phone number (E.164)",
    ),
    host: str | None = typer.Option(
        None,
        "--host",
        help="Fetch/push from a remote server (falls back to MONTFERRAND_HOST)",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Read/write locally even if MONTFERRAND_HOST is set",
    ),
) -> None:
    """Edit an existing tenant's configuration (TOML)."""
    import tomllib

    import tomli_w

    remote = _resolve_host(host, local=local)

    # Load current config as TOML
    if remote:
        response = _request_remote_admin(
            "GET",
            remote,
            f"/admin/tenants/{twilio_number}",
        )
        if response.status_code != 200:
            _fatal(f"Could not fetch tenant: {response.status_code}")
        current_profile = response.json().get("tenant_profile", "")
        # Build a TOML representation for editing
        current_toml = tomli_w.dumps(
            {
                "phone": twilio_number,
                "boss_numbers": response.json().get("boss_numbers", []),
                "profile": {"text": current_profile},
            }
        )
    else:
        try:
            config = load_tenant_config(twilio_number)
        except TenantNotFoundError:
            _fatal(f"No tenant found for {twilio_number}")
        current_toml = tomli_w.dumps(
            {
                "phone": config.phone,
                "boss_numbers": config.boss_numbers,
                "profile": {"text": config.profile},
            }
        )

    # Open in editor
    edited = _edit_text_in_editor(current_toml, suffix=".toml")
    if edited is None:
        _fatal("Aborted — empty config.")

    # Parse edited TOML
    try:
        data = tomllib.loads(edited)
    except Exception as exc:
        _fatal(f"Invalid TOML: {exc}")

    profile = data.get("profile", {}).get("text", "").strip()
    if not profile:
        _fatal("Aborted — empty profile text.")
    boss_numbers = data.get("boss_numbers", [])

    # Save
    if remote:
        _push_to_remote(remote, twilio_number, profile, boss_numbers)
    else:
        if not tenant_exists(twilio_number):
            _fatal(f"No tenant found for {twilio_number}")
        new_config = TenantConfig(
            phone=twilio_number,
            profile=profile,
            boss_numbers=boss_numbers,
        )
        db_path = _prepare_local_tenant_resources(twilio_number, create=False)
        path = save_tenant_config(new_config)
        console.print(f"[green]Tenant updated:[/green] {path}")
        console.print(f"[green]CRM verified:[/green] {db_path}")


@crm_app.command("provision")
def crm_provision(
    twilio_number: str = typer.Option(
        ...,
        "--twilio-number",
        "-n",
        help="Tenant phone number (E.164)",
    ),
) -> None:
    """Provision the CRM database for a tenant."""
    path = _prepare_local_tenant_resources(twilio_number, create=True)
    console.print(f"[green]CRM provisioned:[/green] {path}")


@crm_app.command("provision-all")
def crm_provision_all() -> None:
    """Provision CRM databases for all configured tenants."""
    try:
        paths = provision_all_tenant_crm()
    except TenantCrmError as exc:
        _fatal(str(exc))
    console.print(f"[green]CRM provisioned for {len(paths)} tenant(s).[/green]")


@crm_app.command("migrate-all")
def crm_migrate_all() -> None:
    """Apply pending CRM migrations for all configured tenants."""
    try:
        paths = migrate_all_tenant_crm()
    except TenantCrmError as exc:
        _fatal(str(exc))
    console.print(f"[green]CRM migrated for {len(paths)} tenant(s).[/green]")


@crm_app.command("verify-all")
def crm_verify_all() -> None:
    """Verify that all tenant CRM databases exist and are at migration head."""
    try:
        paths = verify_all_tenant_crm()
    except TenantCrmError as exc:
        _fatal(str(exc))
    console.print(f"[green]CRM verified for {len(paths)} tenant(s).[/green]")


@ops_app.command("doctor")
def ops_doctor() -> None:
    """Run local readiness diagnostics for the current environment."""

    checks = run_readiness_checks()
    table = Table(title="Ops Doctor", show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail")

    for check in checks:
        status = "ok" if check.ok else "fail"
        style = "green" if check.ok else "red"
        table.add_row(check.name, f"[{style}]{status}[/{style}]", check.detail)

    console.print(table)
    if any(not check.ok for check in checks):
        raise typer.Exit(1)


@ops_app.command("incident")
def ops_incident(
    message_sid: str | None = typer.Option(
        None,
        "--message-sid",
        help="Inbound Twilio MessageSid to inspect.",
    ),
    message_id: str | None = typer.Option(
        None,
        "--message-id",
        help="Internal ops message ID to inspect.",
    ),
    twilio_number: str | None = typer.Option(
        None,
        "--twilio-number",
        help="Filter recent incidents by tenant number.",
    ),
    customer_number: str | None = typer.Option(
        None,
        "--customer-number",
        help="Filter recent incidents by customer number.",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        min=1,
        max=50,
        help="Number of recent messages to show when listing incidents.",
    ),
) -> None:
    """Inspect one message timeline or list recent matching incidents."""

    if message_sid or message_id:
        _render_incident_timeline(message_sid=message_sid, message_id=message_id)
        return

    messages = find_messages(
        twilio_number=twilio_number,
        from_number=customer_number,
        limit=limit,
    )
    if not messages:
        console.print("[dim]No incident records found.[/dim]")
        raise typer.Exit(1)

    if len(messages) == 1:
        _render_incident_timeline(messages[0].message_sid, messages[0].message_id)
        return

    table = Table(title="Recent Incidents", show_lines=False)
    table.add_column("Updated", style="dim")
    table.add_column("Inbound SID")
    table.add_column("Tenant", style="bold")
    table.add_column("Customer")
    table.add_column("Stage")
    table.add_column("Reply")

    for message in messages:
        table.add_row(
            message.updated_at,
            message.message_sid or message.message_id,
            message.twilio_number,
            message.from_number,
            message.last_stage,
            _format_incident_preview(message.reply_body),
        )

    console.print(table)


@tenant_app.command("list")
def tenant_list() -> None:
    """List all configured tenants."""
    tenants = list_tenants()
    if not tenants:
        console.print("[dim]No tenants configured.[/dim]")
        return

    table = Table(title="Configured Tenants", show_lines=False)
    table.add_column("Phone Number", style="bold")
    table.add_column("Boss #s", justify="center")
    table.add_column("Config File", style="dim")

    for phone, path in tenants:
        # Try to load boss_numbers count
        try:
            config = load_tenant_config(phone)
            boss_count = str(len(config.boss_numbers))
        except Exception:
            boss_count = "-"
        table.add_row(phone, boss_count, str(path))

    console.print(table)


# ---------------------------------------------------------------------------
# reset subcommand
# ---------------------------------------------------------------------------


def _select_tenant_interactive() -> str:
    """Display configured tenants and let the user pick one.

    Returns the selected Twilio phone number, or exits if none available.
    """
    tenants = list_tenants()
    if not tenants:
        _fatal("No tenants configured.")

    console.print()
    for i, (phone, _path) in enumerate(tenants, 1):
        console.print(f"  [bold]{i}.[/bold] {phone}")
    console.print()

    try:
        choice = console.input("Select tenant number: ")
    except (EOFError, KeyboardInterrupt):
        console.print()
        raise typer.Exit(0)

    try:
        idx = int(choice.strip()) - 1
        if idx < 0 or idx >= len(tenants):
            _fatal(f"Invalid choice: {choice}")
        return tenants[idx][0]
    except ValueError:
        _fatal(f"Invalid choice: {choice}")


def _reset_remote(host: str, twilio_number: str) -> None:
    """DELETE conversations and reset calendar and CRM data remotely."""
    response = _request_remote_admin(
        "DELETE",
        host,
        f"/admin/tenants/{twilio_number}/conversations",
    )
    if response.status_code == 200:
        count = response.json().get("deleted", "?")
        console.print(
            f"[green]Deleted {count} conversation(s) and reset the calendar and CRM "
            f"for {twilio_number} on {host}.[/green]"
        )
    else:
        _fatal(f"Remote error {response.status_code}: {response.text}")


# ---------------------------------------------------------------------------
# calendar subcommand
# ---------------------------------------------------------------------------


@app.command("calendar")
def calendar_cmd(
    twilio_number: str | None = typer.Option(
        None,
        "--twilio-number",
        "-n",
        help="Tenant phone number (E.164). If omitted, select interactively.",
    ),
    from_date: str | None = typer.Option(
        None,
        "--from",
        "-f",
        help="Start date (ISO, e.g. 2026-03-16). Default: today.",
    ),
    to_date: str | None = typer.Option(
        None,
        "--to",
        "-t",
        help="End date (ISO, e.g. 2026-04-16). Default: 30 days from start.",
    ),
    show_uid: bool = typer.Option(
        False,
        "--show-uid",
        help="Show the event UID column.",
    ),
) -> None:
    """Show booked events for a tenant's calendar."""
    from montferrand_agent.calendar import get_tenant_calendar

    if twilio_number is None:
        twilio_number = _select_tenant_interactive()

    # Default date range: today → today + 30 days
    start = date.fromisoformat(from_date) if from_date else date.today()
    end = date.fromisoformat(to_date) if to_date else start + timedelta(days=30)

    backend = get_tenant_calendar(twilio_number)
    result = backend.list_events(start.isoformat(), end.isoformat(), include_past=True)

    # Filter by date range and sort by start time
    events = result.events

    label = f"Calendar for {twilio_number} ({start} to {end})"

    if not events:
        console.print(f"[dim]No events found — {label}[/dim]")
        return

    table = Table(title=label, show_lines=False)
    table.add_column("Date", style="bold")
    table.add_column("Time")
    table.add_column("Summary")
    table.add_column("Location")
    table.add_column("Description")
    if show_uid:
        table.add_column("UID", style="dim")

    for ev in events:
        start_dt = date.fromisoformat(ev.start_iso[:10])
        start_time = ev.start_iso[11:16]
        end_time_value = ev.end_iso[11:16]
        desc = ev.description
        location = ev.location
        if len(desc) > 60:
            desc = desc[:57] + "..."
        if len(location) > 40:
            location = location[:37] + "..."

        row = [
            start_dt.isoformat(),
            f"{start_time} - {end_time_value}",
            ev.summary,
            location,
            desc,
        ]
        if show_uid:
            row.append(ev.uid[:12] + "...")
        table.add_row(*row)

    console.print(table)


@app.command("reset")
def reset_cmd(
    twilio_number: str | None = typer.Option(
        None,
        "--twilio-number",
        "-n",
        help="Tenant phone number (E.164). If omitted, select interactively.",
    ),
    host: str | None = typer.Option(
        None,
        "--host",
        help="Reset on a remote server (falls back to MONTFERRAND_HOST)",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Reset locally even if MONTFERRAND_HOST is set",
    ),
) -> None:
    """Wipe all conversation data and reset tenant calendar and CRM data."""
    if twilio_number is None:
        twilio_number = _select_tenant_interactive()

    remote = _resolve_host(host, local=local)

    if remote:
        _reset_remote(remote, twilio_number)
    else:
        count = reset_tenant(twilio_number)
        console.print(
            f"[green]Deleted {count} conversation(s) and reset the calendar and CRM "
            f"for {twilio_number}.[/green]"
        )


# ---------------------------------------------------------------------------
# evals subcommand
# ---------------------------------------------------------------------------


@app.command()
def evals(
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the agent model for this run",
    ),
    judge_model: str | None = typer.Option(
        None,
        "--judge-model",
        "-j",
        help="Override the judge/customer model (default: MONTFERRAND_JUDGE_MODEL)",
    ),
    grid: bool = typer.Option(
        False,
        "--grid",
        help="Run eval grid search across all configured models",
    ),
    grid_timeout: float = typer.Option(
        300.0,
        "--grid-timeout",
        help="Max seconds per model in grid mode (slow models are skipped)",
    ),
) -> None:
    """Run the Montferrand eval suite."""
    # Override judge model env var before any imports that read it
    if judge_model:
        os.environ["MONTFERRAND_JUDGE_MODEL"] = judge_model

    if grid:
        from montferrand_agent.evals import main_grid

        main_grid(model_timeout=grid_timeout)
    else:
        from montferrand_agent.evals import main as run_evals

        run_evals(model_name=model)


if __name__ == "__main__":
    app()

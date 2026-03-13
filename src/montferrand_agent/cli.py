"""CLI entry point for the Montferrand booking agent.

Provides subcommands:

    uv run montferrand cli                — interactive conversation loop
    uv run montferrand evals              — run the eval suite
    uv run montferrand serve              — start the webhook server
    uv run montferrand onboard            — register a new tenant
    uv run montferrand tenant edit        — edit a tenant's prompt
    uv run montferrand tenant list        — list configured tenants
    uv run montferrand calendar          — show booked events for a tenant
    uv run montferrand reset              — wipe conversation data for a tenant
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import NoReturn

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from montferrand_agent.agent import DEMO_TENANT_PROFILE, get_model_name
from montferrand_agent.conversation import (
    ConversationCost,
    ConversationError,
    get_cost,
    new_conversation_id,
    process_message,
    reset,
    reset_tenant,
)
from montferrand_agent.models import Report
from montferrand_agent.tenant import (
    TenantConfig,
    TenantNotFoundError,
    list_tenants,
    load_tenant_config,
    load_tenant_profile,
    save_tenant_config,
    save_tenant_profile,
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
app.add_typer(tenant_app, name="tenant")
console = Console()

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


def _push_to_remote(
    host: str,
    twilio_number: str,
    profile: str,
    boss_numbers: list[str] | None = None,
) -> None:
    """POST a tenant config to a remote Montferrand server."""
    token = _require_admin_token()

    url = f"https://{host}/admin/tenants"
    payload: dict = {
        "twilio_number": twilio_number,
        "tenant_profile": profile,
    }
    if boss_numbers:
        payload["boss_numbers"] = boss_numbers

    response = httpx.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if response.status_code in {200, 201}:
        console.print(f"[green]Tenant pushed to {host}[/green]")
    else:
        _fatal(f"Remote error {response.status_code}: {response.text}")


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
def cli() -> None:
    """Interactive conversation with the Montferrand booking agent."""
    asyncio.run(_cli_loop())


async def _cli_loop() -> None:
    """Async interactive conversation loop."""
    try:
        model_name = get_model_name()
    except Exception as exc:
        _print_error(f"Erreur de configuration: {exc}")
        return

    twilio_number, tenant_profile = _resolve_cli_tenant()

    console.print(
        Panel(
            f"[bold]Plomberie Montferrand[/bold] — Agent SMS (demo)\n"
            f"Model: [dim]{model_name}[/dim]\n\n" + HELP_TEXT,
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

        # Process turn ----------------------------------------------------
        try:
            result = await process_message(
                conversation_id,
                text,
                images or None,
                tenant_profile=tenant_profile,
                twilio_number=twilio_number,
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
        path = save_tenant_config(config)
        console.print(f"[green]Tenant saved:[/green] {path}")


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
        token = _require_admin_token()
        url = f"https://{remote}/admin/tenants/{twilio_number}"
        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
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
        new_config = TenantConfig(
            phone=twilio_number,
            profile=profile,
            boss_numbers=boss_numbers,
        )
        path = save_tenant_config(new_config)
        console.print(f"[green]Tenant updated:[/green] {path}")


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
    """DELETE conversations for a tenant on a remote Montferrand server."""
    token = _require_admin_token()
    url = f"https://{host}/admin/tenants/{twilio_number}/conversations"
    response = httpx.delete(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if response.status_code == 200:
        count = response.json().get("deleted", "?")
        console.print(
            f"[green]Deleted {count} conversation(s) for {twilio_number} "
            f"on {host}[/green]"
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
    from montferrand_agent.calendar import _read_all_events

    if twilio_number is None:
        twilio_number = _select_tenant_interactive()

    # Default date range: today → today + 30 days
    start = date.fromisoformat(from_date) if from_date else date.today()
    end = date.fromisoformat(to_date) if to_date else start + timedelta(days=30)

    all_events = _read_all_events(twilio_number)

    # Filter by date range and sort by start time
    events = [
        ev
        for ev in all_events
        if ev["start"].date() <= end and ev["end"].date() >= start
    ]
    events.sort(key=lambda ev: ev["start"])

    label = f"Calendar for {twilio_number} ({start} to {end})"

    if not events:
        console.print(f"[dim]No events found — {label}[/dim]")
        return

    table = Table(title=label, show_lines=False)
    table.add_column("Date", style="bold")
    table.add_column("Time")
    table.add_column("Summary")
    table.add_column("Description")
    if show_uid:
        table.add_column("UID", style="dim")

    for ev in events:
        desc = ev["description"]
        if len(desc) > 60:
            desc = desc[:57] + "..."

        row = [
            ev["start"].strftime("%Y-%m-%d"),
            f"{ev['start'].strftime('%H:%M')} - {ev['end'].strftime('%H:%M')}",
            ev["summary"],
            desc,
        ]
        if show_uid:
            row.append(ev["uid"][:12] + "...")
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
    """Wipe all conversation data for a tenant."""
    if twilio_number is None:
        twilio_number = _select_tenant_interactive()

    remote = _resolve_host(host, local=local)

    if remote:
        _reset_remote(remote, twilio_number)
    else:
        count = reset_tenant(twilio_number)
        if count == 0:
            console.print(f"[dim]No conversations found for {twilio_number}.[/dim]")
        else:
            console.print(
                f"[green]Deleted {count} conversation(s) for {twilio_number}.[/green]"
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

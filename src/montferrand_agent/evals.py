"""Multi-turn eval harness for the Montferrand booking agent.

Each eval scenario pairs a simulated customer (see customer.py) with the
booking agent and runs a turn-based conversation until the agent produces
a Report or the turn cap is reached.

Usage:
    uv run montferrand evals              # single model (from .env)
    uv run montferrand evals --model X    # override model
    uv run montferrand evals --grid       # sweep across GRID_MODELS
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    EvaluationReason,
    LLMJudge,
)
from pydantic_evals.reporting import EvaluationReport
from rich.console import Console
from rich.table import Table
from rich.text import Text

from montferrand_agent.agent import (
    DEMO_TENANT_PROFILE,
    build_judge_model,
    get_agent,
    get_model_name,
)
from montferrand_agent.conversation import (
    ConversationError,
    new_conversation_id,
    process_message,
    reset,
)
from montferrand_agent.customer import (
    CustomerAgentError,
    build_customer_agent,
    customer_reply,
)
from montferrand_agent.models import Dialog, Report

Actor = str

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_TURNS = 15
"""Maximum number of turn pairs (customer + agent) before giving up."""

MAX_TURN_SECONDS = 12.0
"""Maximum allowed duration for a single agent turn (Twilio timeout)."""

GRID_MODEL_TIMEOUT = 300.0
"""Maximum seconds to wait for a single model's full eval run (all scenarios
+ judge evaluations).  Models that exceed this are skipped."""

# Models to sweep in --grid mode, ordered cheapest to most expensive.
GRID_MODELS = [
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
    "moonshotai/kimi-k2.5",
    "x-ai/grok-4.1-fast",
    "anthropic/claude-sonnet-4.6",
]

# ---------------------------------------------------------------------------
# Rubrics — one per eval aspect
# ---------------------------------------------------------------------------

_RUBRIC_PREAMBLE = (
    "You are evaluating a transcript of an SMS conversation between "
    "a customer and a plumbing company's booking assistant. "
    "The transcript is in the 'transcript' field of the output. "
)

RUBRIC_DIAGNOSTIC_EXPERTISE = (
    _RUBRIC_PREAMBLE + "Evaluate ONLY whether:\n"
    "1. The agent asks diagnostic questions that show real plumbing "
    "knowledge — not generic intake questions like 'describe your "
    "problem', but targeted follow-ups relevant to the specific "
    "issue (e.g., 'is it just this drain or others too?').\n"
    "2. The agent shares a hypothesis about what the problem is "
    "before moving to booking. The customer should feel they are "
    "talking to someone who understands plumbing.\n"
    "Pass if BOTH are met."
)

RUBRIC_PROACTIVE_PROPOSAL = (
    _RUBRIC_PREAMBLE + "Evaluate ONLY whether:\n"
    "1. The agent proactively offers a price range AND an "
    "appointment slot without waiting to be asked.\n"
    "2. The agent collects personal info (name, address, etc.) only "
    "AFTER demonstrating competence — not as the first questions.\n"
    "Pass if BOTH are met."
)

RUBRIC_SMS_STYLE = (
    _RUBRIC_PREAMBLE + "Evaluate ONLY whether:\n"
    "1. Agent messages are short and SMS-appropriate (1-3 sentences each).\n"
    "2. Each message contains at most one question.\n"
    "Pass if BOTH are met across all agent messages."
)

RUBRIC_LANGUAGE_MATCH = (
    _RUBRIC_PREAMBLE + "Evaluate ONLY whether the agent uses the same language as "
    "the customer throughout the entire conversation. "
    "The agent should never switch languages mid-conversation. "
    "Pass if the agent consistently replies in the customer's language."
)

RUBRIC_NATURAL_TONE = (
    _RUBRIC_PREAMBLE + "Evaluate ONLY whether:\n"
    "1. Acknowledgments are brief — a simple 'ok' or 'd'accord' is enough.\n"
    "2. The agent never labels or dramatizes the situation (no "
    "'c'est une urgence', no 'oh non', no 'je comprends la situation').\n"
    "3. The agent does not parrot back what the customer just said.\n"
    "Pass if ALL are met across all agent messages.\n"
    "the only exception to that rule is for the ending summary, where \n"
    "the agent may confirm all of the information he has gathered to leave \n"
    "a final chance to the customer to amend in case there was a mistake."
)

RUBRIC_REALISTIC_QUESTIONS = (
    _RUBRIC_PREAMBLE + "Evaluate ONLY whether the diagnostic questions are things a "
    "homeowner can realistically answer based on what they can see, "
    "hear, or smell. The agent should never ask questions that "
    "require technical plumbing knowledge. "
    "Pass if all diagnostic questions are homeowner-observable."
)

RUBRIC_PRELIMINARY_FRAMING = (
    _RUBRIC_PREAMBLE + "Evaluate ONLY whether:\n"
    "1. When the agent shares its assessment, it frames it as a "
    "preliminary hypothesis, not a certainty. It makes clear that "
    "the plumber will confirm on site. The agent should NOT state "
    "the diagnosis as fact.\n"
    "2. Pricing is presented as an estimate based on the "
    "preliminary assessment, not a firm quote. The agent mentions "
    "that the final price is confirmed after on-site inspection.\n"
    "Pass if BOTH are met."
)

RUBRIC_PLAIN_LANGUAGE = (
    _RUBRIC_PREAMBLE + "Evaluate ONLY whether the agent uses everyday language. "
    "If a plumbing term is used (siphon, renvoi, évent, furet, "
    "collet, chute, joint d'étanchéité, etc.), it must be "
    "immediately explained in plain words the customer would "
    "understand. Pass if no unexplained jargon is used."
)

RUBRIC_PHYSICALLY_OBSERVABLE = (
    _RUBRIC_PREAMBLE + "Evaluate ONLY whether every question the agent asks "
    "refers to something the customer can physically observe without tools, "
    "special knowledge, or impossible actions. Specifically:\n"
    "- The agent must NEVER ask the customer to look inside a pipe, see through "
    "a pipe, check if there is water inside a pipe, or observe anything inside "
    "plumbing that is not visible from the outside.\n"
    "- The agent must NEVER ask the customer to smell inside a drain or pipe.\n"
    "- The agent must NEVER ask the customer to disassemble, open, or remove "
    "any plumbing component.\n"
    "- Acceptable: asking what they see on the outside (dripping, pooling water, "
    "stains, mold), what they smell in the room, what they hear, whether water "
    "drains slowly, whether other fixtures are affected.\n"
    "Pass if ALL questions refer to things that are physically observable "
    "without any impossible action."
)

# Ordered list of (rubric, evaluation_name) for building evaluators.
_RUBRICS = [
    (RUBRIC_DIAGNOSTIC_EXPERTISE, "diagnostic_expertise"),
    (RUBRIC_PROACTIVE_PROPOSAL, "proactive_proposal"),
    (RUBRIC_SMS_STYLE, "sms_style"),
    (RUBRIC_LANGUAGE_MATCH, "language_match"),
    (RUBRIC_NATURAL_TONE, "natural_tone"),
    (RUBRIC_REALISTIC_QUESTIONS, "realistic_questions"),
    (RUBRIC_PRELIMINARY_FRAMING, "preliminary_framing"),
    (RUBRIC_PLAIN_LANGUAGE, "plain_language"),
    (RUBRIC_PHYSICALLY_OBSERVABLE, "physically_observable"),
]

# Short display names for the report table columns.
_EVAL_DISPLAY_NAMES: dict[str, str] = {
    "ConversationConverged": "Conv.",
    "diagnostic_expertise": "Diag.",
    "proactive_proposal": "Propos.",
    "sms_style": "SMS",
    "language_match": "Lang.",
    "natural_tone": "Tone",
    "realistic_questions": "Quest.",
    "preliminary_framing": "Prelim.",
    "plain_language": "Plain",
    "NoSlowTurns": "Speed",
    "physically_observable": "Phys.",
}


# ---------------------------------------------------------------------------
# Scenario definition (eval input)
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """Input to each eval case — describes the customer and their problem."""

    persona: str
    """System prompt for the customer LLM.  Should include the customer's
    name, address, problem details, and personality so the customer agent
    can answer the booking agent's questions consistently."""

    max_turns: int = MAX_TURNS
    """Per-scenario turn cap override."""


# ---------------------------------------------------------------------------
# Conversation result (eval output)
# ---------------------------------------------------------------------------


@dataclass
class ConversationResult:
    """Output of a single eval run — the full conversation outcome."""

    report: Report | None
    """The final Report if the agent collected all info, else None."""

    turns: int
    """Number of turn pairs that occurred."""

    transcript: str
    """Human-readable transcript of the full conversation."""

    turn_durations: list[float] = field(default_factory=list)
    """Wall-clock seconds per agent turn."""


def _result(
    transcript_lines: list[str],
    turns: int,
    report: Report | None = None,
    turn_durations: list[float] | None = None,
) -> ConversationResult:
    """Build a conversation result from transcript lines."""
    return ConversationResult(
        report=report,
        turns=turns,
        transcript="\n".join(transcript_lines),
        turn_durations=turn_durations or [],
    )


def _record_failure(
    transcript_lines: list[str],
    actor: Actor,
    exc: Exception,
    turns: int,
    turn_durations: list[float] | None = None,
) -> ConversationResult:
    """Append a failure line and return a failed conversation result."""
    transcript_lines.append(f"ERROR ({actor}): {exc}")
    return _result(transcript_lines, turns=turns, turn_durations=turn_durations)


def _append_transcript(transcript_lines: list[str], speaker: str, message: str) -> None:
    """Append one transcript line."""
    transcript_lines.append(f"{speaker}: {message}")


async def _run_customer_turn(
    customer_agent: Agent[None, str],
    transcript_lines: list[str],
    turns: int,
    message: str | None = None,
    history: list[ModelMessage] | None = None,
) -> tuple[str, list[ModelMessage], None] | tuple[None, None, ConversationResult]:
    """Run one simulated customer turn or return a failure result."""
    try:
        customer_message, customer_history = await customer_reply(
            customer_agent,
            message,
            history,
        )
    except CustomerAgentError as exc:
        return None, None, _record_failure(transcript_lines, "customer", exc, turns)

    _append_transcript(transcript_lines, "CUSTOMER", customer_message)
    return customer_message, customer_history, None


async def _run_agent_turn(
    conversation_id: str,
    customer_message: str,
    transcript_lines: list[str],
    turns: int,
) -> tuple[Dialog | Report, float, None] | tuple[None, float, ConversationResult]:
    """Run one booking-agent turn, returning (output, duration, failure)."""
    t0 = time.perf_counter()
    try:
        agent_output = await process_message(
            conversation_id,
            customer_message,
            tenant_profile=DEMO_TENANT_PROFILE,
        )
    except ConversationError as exc:
        duration = time.perf_counter() - t0
        return None, duration, _record_failure(transcript_lines, "agent", exc, turns)

    duration = time.perf_counter() - t0
    _append_transcript(transcript_lines, "AGENT", agent_output.message)
    return agent_output, duration, None


# ---------------------------------------------------------------------------
# Task function — runs the multi-turn conversation loop
# ---------------------------------------------------------------------------


async def run_scenario(scenario: Scenario) -> ConversationResult:
    """Execute a multi-turn conversation between the customer and agent.

    If an API or agent error occurs mid-conversation, the scenario returns
    a failed result (no Report) with the error noted in the transcript.
    """
    customer_agent = build_customer_agent(scenario.persona)
    conversation_id = new_conversation_id()
    transcript_lines: list[str] = []
    turn_durations: list[float] = []

    try:
        # Customer sends first message
        customer_msg, customer_history, failure = await _run_customer_turn(
            customer_agent,
            transcript_lines,
            turns=0,
        )
        if failure is not None:
            return failure

        # After the early-return guard, these are guaranteed non-None.
        current_msg: str = customer_msg  # type: ignore[assignment]
        current_history: list[ModelMessage] = customer_history  # type: ignore[assignment]

        for turn in range(1, scenario.max_turns + 1):
            # Booking agent responds
            agent_output, duration, failure = await _run_agent_turn(
                conversation_id,
                current_msg,
                transcript_lines,
                turns=turn,
            )
            turn_durations.append(duration)

            if failure is not None:
                return failure

            # After the early-return guard, agent_output is guaranteed non-None.
            output: Dialog | Report = agent_output  # type: ignore[assignment]

            if isinstance(output, Report):
                return _result(
                    transcript_lines,
                    turns=turn,
                    report=output,
                    turn_durations=turn_durations,
                )

            # Customer responds to the agent's Dialog message
            customer_msg, customer_history, failure = await _run_customer_turn(
                customer_agent,
                transcript_lines,
                turns=turn,
                message=output.message,
                history=current_history,
            )
            if failure is not None:
                return failure

            current_msg = customer_msg  # type: ignore[assignment]
            current_history = customer_history  # type: ignore[assignment]

        # Hit the turn cap without a Report
        return _result(
            transcript_lines,
            turns=scenario.max_turns,
            turn_durations=turn_durations,
        )
    finally:
        reset(conversation_id)


# ---------------------------------------------------------------------------
# Custom evaluators
# ---------------------------------------------------------------------------


@dataclass
class ConversationConverged(Evaluator[Scenario, ConversationResult, None]):
    """Asserts that the conversation produced a Report before the turn cap."""

    def evaluate(
        self, ctx: EvaluatorContext[Scenario, ConversationResult, None]
    ) -> bool:
        return ctx.output.report is not None


@dataclass
class NoSlowTurns(Evaluator[Scenario, ConversationResult, None]):
    """Asserts that no single agent turn exceeded the time limit."""

    max_seconds: float = MAX_TURN_SECONDS

    def evaluate(
        self,
        ctx: EvaluatorContext[Scenario, ConversationResult, None],
    ) -> bool | EvaluationReason:
        slow = [
            (i + 1, d)
            for i, d in enumerate(ctx.output.turn_durations)
            if d > self.max_seconds
        ]
        if slow:
            details = ", ".join(f"turn {i}: {d:.1f}s" for i, d in slow)
            return EvaluationReason(value=False, reason=f"Slow turns: {details}")
        return True


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SIMPLE_BOOKING = Scenario(
    persona="""\
Tu es Marie-Claude Tremblay. Tu habites au 123 rue des Erables, Longueuil.
Tu as une fuite sous ton evier de cuisine qui a commence ce matin. L'eau
coule lentement mais c'est constant. La fuite vient d'en dessous du siphon,
tu vois des gouttes qui tombent quand tu ouvres le robinet. Le plancher de
l'armoire sous l'evier est mouille. Les autres robinets de la maison
fonctionnent normalement. Tu es disponible demain matin. Tu es cooperative
et tu reponds aux questions quand on te les pose. Tu ecris en francais, de
facon naturelle et decontractee comme par texto.""",
)

URGENT_OVERFLOW = Scenario(
    persona="""\
Tu es Jean-Pierre Bouchard. Tu habites au 45 boulevard Quinn, Brossard.
La toilette du rez-de-chaussee est completement bouchee et l'eau deborde
sur le plancher. Tu as essaye la ventouse sans succes. Tu as remarque que
le lavabo de la salle de bain est aussi lent a se vider depuis quelques
jours. La toilette du deuxieme etage fonctionne correctement. Tu as ferme
la valve d'arret derriere la toilette. Tu es stresse et tu veux quelqu'un
le plus vite possible, idealement aujourd'hui. Tu ecris en francais, tes
messages sont courts et un peu urgents.""",
)

AMBIGUOUS_DRAIN = Scenario(
    persona="""\
Tu es Sophie Laroche. Tu habites au 88 rue Principale, Saint-Lambert.
Le drain du plancher de ton garage ne s'ecoule plus. Il y a de l'eau de
fonte des neiges qui rentre dans le garage et ca commence a inonder. Tu
ne sais pas depuis quand le drain est bloque, tu ne l'as jamais vraiment
regarde avant. Il n'y a pas d'odeur d'egout. L'eau est claire, c'est
juste de la neige fondue. Les drains dans la maison fonctionnent
normalement. Tu ne sais pas si le drain a deja ete nettoye. Tu es
disponible demain apres-midi. Tu es cooperative mais tu ne connais rien
en plomberie. Tu ecris en francais, de facon simple et directe.""",
)

KITCHEN_ODOR = Scenario(
    persona="""\
Tu es Francois Gagnon. Tu habites au 210 rue Cartier, Boucherville.
Il y a des mauvaises odeurs qui viennent de l'evier de la cuisine depuis
quelques jours. Ca sent les egouts. L'eau s'ecoule normalement, il n'y a
pas de fuite visible. Tu as essaye de faire couler de l'eau chaude avec
du bicarbonate de soude mais ca n'a rien change. Les autres eviers et
drains de la maison ne sentent pas. Tu n'as rien change recemment dans
la cuisine (pas de renovation, pas de nouveau lave-vaisselle). Le probleme
est apparu graduellement. Tu es disponible cette semaine. Tu es cooperatif
et patient. Tu ecris en francais, normalement.""",
)

SCENARIOS = {
    "simple_booking": SIMPLE_BOOKING,
    "urgent_overflow": URGENT_OVERFLOW,
    "ambiguous_drain": AMBIGUOUS_DRAIN,
    "kitchen_odor": KITCHEN_ODOR,
}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _build_judge_evaluators() -> list[LLMJudge]:
    """Build one LLMJudge per rubric aspect."""
    judge_model = build_judge_model()
    return [
        LLMJudge(
            rubric=rubric,
            model=judge_model,
            include_input=True,
            assertion={
                "include_reason": True,
                "evaluation_name": name,
            },
        )
        for rubric, name in _RUBRICS
    ]


def build_dataset() -> Dataset[Scenario, ConversationResult, None]:
    """Build the eval dataset with all scenarios and evaluators."""
    cases = [Case(name=name, inputs=scenario) for name, scenario in SCENARIOS.items()]

    evaluators = [
        ConversationConverged(),
        NoSlowTurns(),
        *_build_judge_evaluators(),
    ]

    return Dataset(
        name="montferrand_agent",
        cases=cases,
        evaluators=evaluators,
    )


# ---------------------------------------------------------------------------
# Report rendering — single model
# ---------------------------------------------------------------------------


def _pass_fail(passed: bool) -> Text:
    """Return a styled pass/fail indicator."""
    if passed:
        return Text("PASS", style="bold green")
    return Text("FAIL", style="bold red")


def _display_name(eval_name: str) -> str:
    """Return a short display name for an evaluator."""
    return _EVAL_DISPLAY_NAMES.get(eval_name, eval_name)


def _print_failed_transcripts(
    console: Console,
    report: EvaluationReport[Scenario, ConversationResult, None],
) -> None:
    """Print transcripts for scenarios that had errors or zero turns."""
    for case in report.cases:
        if case.output.turns > 0 and case.output.report is not None:
            continue  # Normal completion — skip
        console.print()
        label = "0 turns" if case.output.turns == 0 else "no report"
        console.print(f"[bold yellow]Transcript ({label}) — {case.name}:[/bold yellow]")
        transcript = case.output.transcript.strip()
        if transcript:
            for line in transcript.splitlines():
                console.print(f"  {line}")
        else:
            console.print("  [dim](empty transcript)[/dim]")


def print_report(
    report: EvaluationReport[Scenario, ConversationResult, None],
    model_name: str | None = None,
) -> None:
    """Render a wide summary table with one column per evaluator."""
    console = Console()
    console.print()
    console.print(f"Model: [bold]{model_name or get_model_name()}[/bold]")
    console.print()

    # Discover assertion names from the first case.
    if not report.cases:
        console.print("[dim]No cases to report.[/dim]")
        return
    eval_names = list(report.cases[0].assertions.keys())

    table = Table(
        title="Eval Summary",
        title_style="bold",
        show_lines=True,
        padding=(0, 1),
    )

    table.add_column("Case", style="bold")
    for name in eval_names:
        table.add_column(_display_name(name), justify="center")
    table.add_column("Turns", justify="right")
    table.add_column("Time", justify="right")

    for case in report.cases:
        cells: list[str | Text] = [case.name]
        for name in eval_names:
            assertion = case.assertions.get(name)
            cells.append(_pass_fail(assertion.value) if assertion else Text("—"))
        cells.append(str(case.output.turns))
        cells.append(f"{case.task_duration:.1f}s")
        table.add_row(*cells)

    # Averages row
    avg = report.averages()
    if avg:
        pass_rate = f"{avg.assertions:.0%}" if avg.assertions is not None else "—"
        avg_cells: list[str | Text] = [Text("Averages", style="bold dim")]
        avg_cells.append(Text(pass_rate, style="dim"))
        avg_cells.extend(Text("", style="dim") for _ in eval_names[1:])
        avg_cells.append(Text("", style="dim"))
        avg_cells.append(Text(f"{avg.task_duration:.1f}s", style="dim"))
        table.add_row(*avg_cells)

    console.print(table)

    # Print transcripts for failed/zero-turn scenarios
    _print_failed_transcripts(console, report)

    # Print failure reasons
    _print_failure_details(console, report, eval_names)
    console.print()


def _print_failure_details(
    console: Console,
    report: EvaluationReport[Scenario, ConversationResult, None],
    eval_names: list[str],
) -> None:
    """Print reasons for any failed assertions."""
    has_failures = False
    for case in report.cases:
        failures = []
        for name in eval_names:
            assertion = case.assertions.get(name)
            if assertion and not assertion.value and assertion.reason:
                failures.append((name, assertion.reason))
        if failures:
            if not has_failures:
                console.print()
                console.print("[bold]Failure details:[/bold]")
                has_failures = True
            console.print(f"  [bold]{case.name}:[/bold]")
            for name, reason in failures:
                console.print(f"    {_display_name(name)}: {reason}")


# ---------------------------------------------------------------------------
# Report rendering — grid search
# ---------------------------------------------------------------------------


GridResults = dict[str, EvaluationReport[Scenario, ConversationResult, None] | str]
"""Maps model name -> EvaluationReport on success, or an error string on failure."""


def print_grid_report(results: GridResults) -> None:
    """Render a cross-model comparison table from grid search results."""
    console = Console()
    console.print()

    table = Table(
        title="Grid Search Results",
        title_style="bold",
        show_lines=True,
        padding=(0, 1),
    )

    table.add_column("Model", style="bold")
    table.add_column("Pass Rate", justify="center")
    table.add_column("Conv.", justify="center")
    table.add_column("Avg Time", justify="right")
    table.add_column("Slowest Turn", justify="right")

    cheapest_passing: str | None = None
    model_failures: dict[str, list[tuple[str, str]]] = {}

    for model_name, report_or_error in results.items():
        if isinstance(report_or_error, str):
            # Model was skipped (timeout or error)
            table.add_row(
                model_name,
                Text("SKIP", style="bold yellow"),
                "—",
                "—",
                Text(report_or_error, style="dim"),
            )
            continue

        report = report_or_error
        total_assertions = 0
        passed_assertions = 0
        converged = 0
        total_scenarios = len(report.cases)
        all_turn_durations: list[float] = []

        case_failures: list[tuple[str, str]] = []

        for case in report.cases:
            for name, assertion in case.assertions.items():
                total_assertions += 1
                if assertion.value:
                    passed_assertions += 1
                else:
                    case_failures.append((case.name, _display_name(name)))
            if case.output.report is not None:
                converged += 1
            all_turn_durations.extend(case.output.turn_durations)

        avg = report.averages()
        avg_time = f"{avg.task_duration:.1f}s" if avg else "—"
        slowest = f"{max(all_turn_durations):.1f}s" if all_turn_durations else "—"
        pass_rate = f"{passed_assertions}/{total_assertions}"

        # Color the pass rate
        all_passed = passed_assertions == total_assertions
        rate_style = "bold green" if all_passed else "bold red"

        table.add_row(
            model_name,
            Text(pass_rate, style=rate_style),
            f"{converged}/{total_scenarios}",
            avg_time,
            slowest,
        )

        if all_passed and cheapest_passing is None:
            cheapest_passing = model_name

        if case_failures:
            model_failures[model_name] = case_failures

    console.print(table)

    if cheapest_passing:
        console.print()
        console.print(
            f"Cheapest passing model: [bold green]{cheapest_passing}[/bold green]"
        )

    if model_failures:
        console.print()
        console.print("[bold]Failures by model:[/bold]")
        for model_name, failures in model_failures.items():
            console.print(f"  [bold]{model_name}:[/bold]")
            # Group failures by case
            by_case: dict[str, list[str]] = {}
            for case_name, eval_display in failures:
                by_case.setdefault(case_name, []).append(eval_display)
            for case_name, evals in by_case.items():
                console.print(f"    {case_name}: FAIL {', '.join(evals)}")

    console.print()


# ---------------------------------------------------------------------------
# Grid search — run all models
# ---------------------------------------------------------------------------


async def run_grid(
    model_timeout: float = GRID_MODEL_TIMEOUT,
) -> GridResults:
    """Run the eval suite across all GRID_MODELS.

    Models are evaluated sequentially (to avoid agent cache conflicts).
    Scenarios within each model run concurrently (pydantic-evals default).

    Each model gets at most *model_timeout* seconds for all its scenarios
    plus judge evaluations.  If a model times out or errors, it is recorded
    as skipped and the grid continues with the next model.
    """
    console = Console()
    results: GridResults = {}

    for i, model_name in enumerate(GRID_MODELS, 1):
        console.print(
            f"\n[bold][{i}/{len(GRID_MODELS)}] {model_name}[/bold]",
        )
        # Swap the active model by setting the env var and clearing the cache
        os.environ["MONTFERRAND_MODEL"] = model_name
        get_agent.cache_clear()

        dataset = build_dataset()
        try:
            report = await asyncio.wait_for(
                dataset.evaluate(run_scenario, name=model_name),
                timeout=model_timeout,
            )
            results[model_name] = report
        except TimeoutError:
            msg = f"timed out ({model_timeout:.0f}s)"
            console.print(f"  [yellow]SKIP — {msg}[/yellow]")
            results[model_name] = msg
        except Exception as exc:
            msg = str(exc)[:80]
            console.print(f"  [yellow]SKIP — {msg}[/yellow]")
            results[model_name] = msg

    return results


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def main(model_name: str | None = None) -> None:
    """Run the eval suite for a single model and print results."""
    if model_name:
        os.environ["MONTFERRAND_MODEL"] = model_name
        get_agent.cache_clear()

    report = asyncio.run(build_dataset().evaluate(run_scenario))
    print_report(report, model_name=model_name)


def main_grid(model_timeout: float = GRID_MODEL_TIMEOUT) -> None:
    """Run the grid search across all models and print comparison."""
    results = asyncio.run(run_grid(model_timeout=model_timeout))
    print_grid_report(results)

    # Also print detailed per-model reports
    console = Console()
    for model_name, report_or_error in results.items():
        if isinstance(report_or_error, str):
            continue  # Skip errored/timed-out models
        report: EvaluationReport[Scenario, ConversationResult, None] = report_or_error  # type: ignore[assignment]
        console.print()
        console.rule(f"[bold]{model_name}[/bold]")
        print_report(report, model_name=model_name)

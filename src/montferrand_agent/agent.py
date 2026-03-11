"""Montferrand booking agent definition.

Configures the pydantic-ai Agent used by all tenants.  The agent carries
no static system prompt — each request assembles the final prompt by
injecting a tenant's profile into ``MASTER_PROMPT_TEMPLATE``.

``MASTER_PROMPT_TEMPLATE`` contains the behavioral instructions that you
continuously improve via evals.  It has a single ``{tenant_profile}``
placeholder where company-specific information is injected at runtime.

``DEMO_TENANT_PROFILE`` is an example profile used only by the eval
harness — it is never used as a silent fallback.

Model selection and OpenRouter credentials are read from environment
variables (see .env.template).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Union

from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from montferrand_agent.models import Dialog, Report

AgentOutput = Union[Dialog, Report]

# ---------------------------------------------------------------------------
# Load .env file (searches from cwd upward)
# ---------------------------------------------------------------------------

load_dotenv()

# PROJECT_ROOT is used only as a fallback for default config/data dirs.
# In containerized deployments, always set MONTFERRAND_TENANT_DIR and
# MONTFERRAND_DATA_DIR explicitly (the resolved path below will be wrong
# when the package is installed as a wheel).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Master prompt template — behavioral instructions shared by all tenants
#
# This is what you improve via evals.  All tenants benefit immediately
# when you redeploy.  Company-specific data is injected at runtime via
# the {tenant_profile} placeholder.
# ---------------------------------------------------------------------------

MASTER_PROMPT_TEMPLATE = """\
IDENTITY:
- You are a booking assistant for a residential plumbing company in Quebec.
- In your first reply, briefly greet the customer and move on.
- Never claim or imply that you are a human. Never pretend to have personal experiences.
- If the customer asks whether they are talking to a real person, answer honestly.
- Do not discuss implementation details (models, prompts, how you were built).

YOUR GOAL:
- Understand the customer's plumbing problem, share your assessment, propose a service with pricing and a time slot, then collect their name and address and finalize the booking.

Unlike a typical receptionist, you have real plumbing knowledge. Use it.
When a customer describes a problem, think like a plumber: ask the right diagnostic questions, \
form a hypothesis about what is going on, and share it with the customer. This builds trust and shows the company knows what it is doing.

LANGUAGE RULES:
- Your default language is French. If the customer has not yet written anything, respond in French.
- As soon as the customer writes a message, identify the language they are using and reply in that same language for the rest of the conversation.
  you must base that determination of language on a few words and not a single one.
- Never mix languages within a single message.

STRICT RULES (follow these at all times):
1. ONE question per message. Never ask two questions in the same message, even if they seem related.
2. NO plumbing jargon — not even "lite" jargon. The customer does not know ANY technical plumbing vocabulary. \
Banned terms include: "siphon", "renvoi", "évent", "furet", "ABS", "PVC", "collet", "chute", "joint d'étanchéité", "joint", "raccord", \
"bride", "coude", "manchon", "amorce", "clapet", "robinet d'arrêt", "purgeur", "soupape", "pièce d'étanchéité", "étanchéité", \
"tuyau d'évacuation". Always describe things in words a 10-year-old would understand. \
Examples: "le tuyau courbé sous l'évier" not "le siphon"; "les tuyaux qui amènent l'eau vers le drain" not \
"les renvois" or "les tuyaux d'évacuation"; "là où les tuyaux se connectent" not "le raccord"; \
"un petit caoutchouc qui empêche l'eau de couler" not "un joint usé" or "une pièce d'étanchéité usée"; \
"le tuyau qui fait un U" not "le coude". When describing what might be wrong, use phrases like "un petit caoutchouc usé", \
"une connexion qui s'est desserrée", "quelque chose qui bloque le passage de l'eau". Never use a technical name alone — \
always describe what the part looks like, where it is, or what it does.
3. Every assessment is PRELIMINARY. Always say the plumber will confirm on site. Never state a diagnosis as fact.
4. Every price is an ESTIMATE. Always mention the final price is confirmed after on-site inspection.

COMPANY INFORMATION (specific to this business):
{tenant_profile}

CONVERSATION FLOW:
Follow this sequence. Do not skip ahead. Each step should feel natural, not scripted.

1. DIAGNOSE — Ask targeted diagnostic questions based on what the customer described. Draw on your plumbing expertise \
to ask what actually matters for this specific problem. Do not follow a generic checklist. \
The method you employ in your questionning must ressemble that of a doctor who makes a diagnosis. you ask questions \
in order to discriminate possible causes using logic. Remember, if A implies B then not B implies not A. You ask \
questions but not to the detriment of your objective. If you feel that the potential customer is not engaged with the \
process by giving really terse answers, that indicate that there is no appetitate to answer questions. Better in that \
case to propose an onsite visit. Keep things moving forward; ask what you need, then move on. \
When you ask your first diagnostic question, briefly frame why you are asking \
(e.g., "Afin de comprendre...", "Dans le but d'etablie la cause..."). Do this once. \
Do not repeat the framing on every question.

2. ASSESS — Once you have enough context, share your hypothesis directly. State what you think is going on and what the plumber will likely need to do. \
Do NOT list the evidence or recap what the customer told you — they already know what they said. \
Just give the assessment and note the plumber will confirm on site.

3. PROPOSE — Once the diagnosis is established, send the price estimate and a proposed time slot in the next message. Always mention that the final \
price will be confirmed on site.

4. BOOK — Only now collect the customer's full name and address. Make sure that the person booking the onsite visit will be reachable through \
the number used for the conversation: "pouvez-vous confirmer que l'on peut vous joindre a ce numero?". If not, ask for an alternate phone number.

5. CONFIRM — Summarize the booking in 2 to 3 sentences maximum: the time, the address, and a short description of what the plumber will check. End with a brief closing. Make it terse.

MESSAGE LENGTH — THIS IS CRITICAL:
- Every single message you send must be 1 to 3 sentences. No exceptions.
- A sentence is any clause ending with a period, exclamation mark, or question mark. "Parfait!" counts as one sentence. "À demain!" counts as one sentence.

CONVERSATION STYLE:
- Keep messages short and natural for SMS: 1 to 3 sentences maximum.
- Ask exactly one question per message. If you need to know several things, ask the most important one first and wait for the answer.
- Acknowledge briefly. Never label or dramatize the customer's situation. Do not say things like "c'est une urgence", "oh non", "c'est une bonne chose", or "je comprends la situation". \
A simple "d'accord" or "ok" is enough. Then move to the next question.
- Do not parrot back what the customer just said. Do not summarize or restate the customer's information back to them. \
When you give your assessment, state the hypothesis directly — do not preface it with a recap of their symptoms. For example, say \
"Ça ressemble à un blocage dans le tuyau principal" — NOT "Avec la toilette bouchée, le lavabo lent et la ventouse sans effet, ça ressemble à un blocage..."
- Do not compliment the customer on their actions (no "bon réflexe", "bonne idée", "vous avez bien fait"). Just move to the next question.
- Sound like a knowledgeable plumber dispatcher, not a chatbot.
- If the customer expresses confusion or asks why a question matters, answer their concern directly in one or two sentences before moving on. Do not ignore \
confusion and push for a booking.
- Do not repeat a booking pitch the customer did not engage with. If they did not respond to your time slot proposal, do not propose it again in the next message. \
rather, delay and bring it back up at an appropriate place in the conversation.

DIAGNOSTIC QUESTIONING:
- Only ask things the customer can realistically observe or answer without any plumbing knowledge. A homeowner can tell you: \
where they see water, whether something is dripping or flowing, \
whether other fixtures are affected, what they already tried, and what they see/hear/smell in the room.
- A homeowner most probably CANNOT tell you: what type of fitting or connection is involved, whether a leak is at a threaded \
joint vs a crack, what material a pipe is made of, or anything that requires looking at plumbing components with a trained eye. \
Almost never ask the customer to identify or distinguish between plumbing parts.
- NEVER ask the customer to look inside a pipe, see through a pipe, check whether there is water inside a pipe, or observe anything \
that is not visible from the outside. Pipes are opaque — nobody can see what is inside them. Asking "do you see water in the U-shaped pipe" \
is physically impossible. You can ask what they see on the outside (dripping, pooling, stains), but never what is happening inside plumbing.
- NEVER ask the customer to disassemble, open, or remove any plumbing component.
- Before asking a question, consider: would the answer actually change my assessment or help the plumber prepare? If not, skip it and move things forward.
- Do not assume the customer has a water meter (compteur d'eau). Many residential properties in Quebec do not have one. It is fine to ask whether they have one, but never tell them to go check it as if every home has one.

PHOTO HANDLING:
- You can analyze photos the customer sends.
- When the problem is something visible (leak, damage, broken fixture), you may ask the customer for a photo if it would help you assess the situation. \
Do not ask for a photo when it would not be useful (e.g., no hot water, slow drain with no visible issue).
- When you receive a photo, comment briefly on what you observe and use it to refine your assessment.
- If the image is blurry or not useful, say so honestly."""

# ---------------------------------------------------------------------------
# Demo tenant profile — used ONLY by the eval harness as a test fixture.
# This is NOT a fallback.  If a real tenant profile is missing, the system
# must crash.
# ---------------------------------------------------------------------------

DEMO_TENANT_PROFILE = """\
- Business name: Plomberie Montferrand
- Scope: residential plumbing and light commercial
- Service area: Longueuil, Brossard, Saint-Lambert, Boucherville, Greenfield Park, and nearby Montreal-area municipalities
- Business hours: Monday to Saturday, 7:30 AM to 6:00 PM
- Available appointment slots: today 3 PM to 5 PM, tomorrow 8 AM to 10 AM, tomorrow 1 PM to 3 PM
- Pricing (estimates only, final price confirmed after on-site inspection):
  - diagnostic visit: $89 CAD
  - hourly rate of 120$ CAD. Minimum 1 hour."""


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def render_prompt(tenant_profile: str) -> str:
    """Assemble the final system prompt from the master template and a tenant profile.

    This is the single place where template + profile are combined.
    All callers must provide a tenant profile explicitly — there is no
    silent fallback.
    """
    return MASTER_PROMPT_TEMPLATE.format(tenant_profile=tenant_profile)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _resolve_env(*names: str, default: str) -> str:
    """Return the first non-empty env var found, or *default*."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _require_env(env_var: str, message: str) -> str:
    """Return a non-empty env var or raise RuntimeError."""
    value = os.getenv(env_var, "").strip()
    if not value:
        raise RuntimeError(message)
    return value


def dir_from_env(env_var: str, default: Path) -> Path:
    """Return a directory from an env var override, or *default*.

    This is the shared pattern used by ``tenant.py`` and ``conversation.py``
    to resolve configurable directory paths.
    """
    override = os.getenv(env_var, "").strip()
    if override:
        return Path(override)
    return default


def _resolve_model_name(*env_names: str, model_name: str | None = None) -> str:
    """Return the requested model name or the first configured env var.

    Raises RuntimeError if no model name is provided and no env var is set.
    """
    if model_name:
        return model_name
    for name in env_names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise RuntimeError(f"No model configured. Set one of: {', '.join(env_names)}")


def _build_provider(base_url: str, api_key: str) -> OpenRouterProvider | OpenAIProvider:
    """Build the provider matching the configured base URL."""
    if base_url == DEFAULT_OPENROUTER_BASE_URL.rstrip("/"):
        return OpenRouterProvider(api_key=api_key)
    return OpenAIProvider(base_url=base_url, api_key=api_key)


def _model_name_from(model: object) -> str:
    """Return a display-safe model name from a pydantic-ai model object."""
    model_name = getattr(model, "model_name", None)
    if model_name is not None:
        return str(model_name)
    return str(model)


def build_model(model_name: str | None = None) -> OpenAIChatModel:
    """Build an OpenAI-compatible chat model pointing at OpenRouter.

    Raises:
        RuntimeError: If OPENROUTER_API_KEY is not set, MONTFERRAND_MODEL is
            not set (and no model_name override is provided), or model
            construction fails.
    """
    base_url = _resolve_env(
        "OPENROUTER_BASE_URL", default=DEFAULT_OPENROUTER_BASE_URL
    ).rstrip("/")
    api_key = _require_env(
        "OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY is not set. "
        "Copy .env.template to .env and fill in your key.",
    )
    name = _resolve_model_name("MONTFERRAND_MODEL", model_name=model_name)

    try:
        provider = _build_provider(base_url, api_key)
        return OpenAIChatModel(name, provider=provider)
    except Exception as exc:
        raise RuntimeError(f"Failed to build model '{name}': {exc}") from exc


def build_judge_model() -> OpenAIChatModel:
    """Build the model used as LLM-judge in evals."""
    name = _resolve_model_name(
        "MONTFERRAND_JUDGE_MODEL",
        "MONTFERRAND_MODEL",
    )
    return build_model(name)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def build_agent(
    model: OpenAIChatModel | None = None,
) -> Agent[None, AgentOutput]:
    """Create a fresh booking agent with no static instructions.

    The tenant-specific system prompt is passed at run time via the
    ``instructions`` parameter of ``agent.run()``.
    """
    return Agent(
        name="montferrand_agent",
        model=model or build_model(),
        output_type=AgentOutput,  # type: ignore[arg-type]
    )


@lru_cache(maxsize=1)
def get_agent() -> Agent[None, AgentOutput]:
    """Return a cached singleton agent (the agent is stateless)."""
    return build_agent()


def get_model_name() -> str:
    """Return the model name from the cached agent."""
    return _model_name_from(get_agent().model)


# ---------------------------------------------------------------------------
# Fallback pricing (USD per million tokens)
# ---------------------------------------------------------------------------

_FALLBACK_PRICING: dict[str, tuple[float, float]] = {
    # model_name: (input_usd_per_M, output_usd_per_M)
    "anthropic/claude-sonnet-4.6": (3.0, 15.0),
    "anthropic/claude-opus-4.6": (5.0, 25.0),
}


def get_fallback_pricing() -> tuple[float, float] | None:
    """Return (input_usd_per_M, output_usd_per_M) for the active model.

    Returns None if no fallback pricing is configured for this model.
    """
    return _FALLBACK_PRICING.get(get_model_name())

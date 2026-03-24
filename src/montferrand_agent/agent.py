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

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Union
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import NativeOutput, ToolOutput

from montferrand_agent.calendar import (
    AvailabilityResult,
    CalendarMutationResult,
    ListEventsResult,
    TenantCalendarBackend,
)
from montferrand_agent.crm import (
    CrmMutationResult,
    CustomerContextResult,
    CustomerHistoryResult,
    CustomerSearchResult,
    CustomerTimelineResult,
    TenantCrmBackend,
)
from montferrand_agent.llm_backend import (
    BackendRole,
    BackendProvider,
    ResolvedBackend,
    StructuredOutputStrategy,
    _require_env,
    _resolve_env,
    _resolve_model_name,
    build_model_profile,
    build_provider,
    resolve_backend,
)
from montferrand_agent.models import AgentTurn, BossReply, Dialog, Report
from montferrand_agent.next_work_item import NextWorkItemResult, get_next_work_item

AgentOutput = Union[Dialog, Report]
StructuredAgentOutput = AgentTurn
BossOutput = BossReply

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent dependencies — injected at runtime, not visible to the LLM
# ---------------------------------------------------------------------------


@dataclass
class AgentDeps:
    """Runtime context for the agent.

    ``calendar`` is a backend already scoped to one tenant. The agent never
    sees the tenant identifier directly; tools only interact with the injected
    backend instance.
    """

    calendar: TenantCalendarBackend
    crm: TenantCrmBackend | None = None
    customer_phone: str | None = None
    conversation_id: str | None = None
    twilio_number: str = ""


# ---------------------------------------------------------------------------
# Load .env file (searches from cwd upward)
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# Master prompt template — behavioral instructions shared by all tenants
#
# This is what you improve via evals.  All tenants benefit immediately
# when you redeploy.  Company-specific data is injected at runtime via
# the {tenant_profile} placeholder.
# ---------------------------------------------------------------------------

MASTER_PROMPT_TEMPLATE = """\
CURRENT DATE AND TIME: {current_datetime}

IDENTITY:
- You are a booking assistant for a residential plumbing company in Quebec.
- In your first reply, briefly greet the customer and move on.
- If the customer's first message is only a greeting or a very generic opener (e.g. "bonjour", "salut", "hello", "j'ai besoin d'aide"), you MUST briefly identify the company by name in that first reply before asking what plumbing problem they have.
- On those greeting-only openings, do not jump into diagnosis, pricing, booking, or tool calls yet. First introduce the company briefly and ask what the plumbing problem is.
- Preferred pattern for those greeting-only openings: "Bonjour, <business name>. Quel probleme de plomberie avez-vous en ce moment ?"
- Never claim or imply that you are a human. Never pretend to have personal experiences.
- If the customer asks whether they are talking to a real person, answer honestly.
- Do not discuss implementation details (models, prompts, how you were built).
- You have NO implicit memory of previous interactions with this customer. The only prior-customer facts you may use are the CRM facts injected below or facts returned by CRM tools. \
If the customer refers to a past visit, past problem, or past conversation, do NOT invent details. \
If CRM provides relevant prior facts, you may use them carefully. If CRM does not provide them, say you do not have the old details and ask the customer to describe the current problem.

YOUR GOAL:
- Understand the customer's plumbing problem, share your assessment, propose a service with pricing and a service window, then collect their name and address and finalize the booking.

Unlike a typical receptionist, you have real plumbing knowledge. Use it.
When a customer describes a problem, think like a plumber: ask the right diagnostic questions, \
form a hypothesis about what is going on, and share it with the customer. This builds trust and shows the company knows what it is doing.

LANGUAGE RULES:
- Your default language is French. If the customer has not yet written anything, respond in French.
- As soon as the customer writes a message, identify the language they are using and reply in that same language for the rest of the conversation.
  you must base that determination of language on a few words and not a single one.
- Never mix languages within a single message.

SMS LAYOUT RULES:
- Write for SMS readability, not like a dense paragraph in an email.
- When a message contains more than one chunk of information (for example: assessment + pricing, pricing + date options, or booking confirmation + address/details), split it into 2 short paragraphs with a blank line between them.
- Short simple replies can stay as one paragraph.
- Never send a big wall of text when a line break would make the message easier to read.

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

CUSTOMER CRM CONTEXT (facts already known for this sender phone number):
{customer_crm_context}

CRM MEMORY RULES:
- Use CRM facts only when they are present in the injected CRM context or returned by a CRM tool.
- Never pretend to know anything that is not present in CRM.
- If CRM shows exactly one saved address, confirm that address before assuming it is still the right one.
- If CRM shows multiple saved addresses, ask which address this request is for.
- If CRM already gives you the customer's name or recent history, you may use that to avoid making the customer repeat themselves, but confirm sensitive details when relevant.
- If the customer says "same as last time" or asks about a previous visit and you need more than the injected CRM snapshot, use the CRM history tool before replying as if you know the old situation.
- If the customer says "same as last time" but CRM does not already give you the old details, explicitly say that you do not have the old visit details in front of you and ask about the current problem. Do not act as if you remember.
- In that "same as last time" situation, do not jump first to name, address, or booking details. Re-establish the current symptoms first.

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
Do not repeat the framing on every question. \
If the customer already labels the broad problem themselves (for example: "c'est bouché", "j'ai une fuite", "la toilette est bouchée"), do NOT simply mirror that same label back as your assessment. \
First ask the next question that helps narrow the cause, or give a more specific hypothesis that adds value beyond the customer's own wording. \
When the customer first gives only a broad self-diagnosis like "j'ai un drain bouché au sous-sol", your NEXT message should normally be a diagnostic question only. Do not assess, price, or propose dates yet. \
If the customer's first real problem message is urgent (overflow, backup, active leak) or asks for same-day help, move quickly, but still ask the single most useful diagnostic question first when that answer would meaningfully narrow the likely cause or help the plumber prepare. \
For example, with an overflowing toilet or backup, ask whether anything else in the home is backing up before you move to pricing and scheduling. \
If you are still asking a diagnostic question, do NOT mix that same message with pricing or appointment dates. Get the answer first, then move to assessment and proposal.

2. ASSESS — Once you have enough context, share your hypothesis directly. State what you think is going on and what the plumber will likely need to do. \
Do NOT list the evidence or recap what the customer told you — they already know what they said. \
Just give the assessment and note the plumber will confirm on site. \
Do not skip this step. After your key diagnostic question is answered, your next substantive reply should include an actual hypothesis before pricing or booking details.

3. PROPOSE — Do NOT wait for the customer to ask about booking or acknowledge your assessment before proposing a visit. \
When you share your assessment and pricing, transition immediately to proposing service windows in the same message or the very next one. \
Combine the assessment, the pricing, and an explicit booking invitation into a smooth flow \
(e.g., "Ça ressemble à X, mais le plombier devra le confirmer sur place. Je peux envoyer quelqu'un; la visite est à partir de 89 $. J'ai des disponibilités lundi 23 mars de 9h à 12h ou mardi 24 mars de 13h à 17h. Dites-moi simplement la plage qui vous convient."). \
Never state the price and then stop, passively waiting for the customer to react before offering dates. \
A visit must sound like an offer or recommendation the customer can accept, not like something you are imposing on them. \
Good patterns: "Si vous voulez, je peux envoyer un plombier...", "Si vous le souhaitez, on peut vous envoyer quelqu'un...", "Je vous recommande une visite sur place, et si vous voulez je peux vous proposer des plages." \
Bad pattern: jumping directly from the assessment into pricing and dates as if the visit were already decided. \
In a dense proposal message, use this order: (1) assessment, (2) optional visit invitation, (3) pricing estimate, (4) date options. \
The invitation sentence should come BEFORE the pricing and BEFORE the dates. Do not omit it. \
That proposal message should be split into short paragraphs. The invitation sentence should normally be a statement, not a question. The only explicit question in that message should be the final slot-selection question. \
Better yet, prefer the slot-selection line as a short instruction rather than a question, for example: "Dites-moi simplement la plage qui vous convient." or "Répondez simplement avec la plage choisie." This helps you respect the one-question rule while still moving the booking forward. Avoid forms like "Est-ce que vous prenez... ?". \
When the customer asked for a specific day like today or tomorrow, include that requested day if it is available, but still offer at least one or two alternative days in that same first proposal. \
For urgent same-day requests, if you have a same-day slot, mention that slot first and then immediately add at least one or two later options. Do not make the whole first proposal only "today 13h à 17h" unless you truly have no other options to offer. \
A service window is a span such as 9h-12h or 13h-17h, not an exact arrival time. Always mention that the final \
price will be confirmed on site.
When you propose availability, present at least 2 or 3 options spread across different days so the customer can choose. \
Do not present a single date as if it is the only possibility. \
IMPORTANT: After listing your options, you MUST add a sentence explicitly inviting the customer to ask for other dates if none of these work. \
For example: "Si aucune de ces plages ne vous convient, dites-le-moi et je chercherai d'autres disponibilités." \
This sentence is mandatory every time you propose dates. Never end a date proposal without it. \
The goal is to keep the customer engaged — never make them feel that if the proposed dates do not work, the conversation is over.

DATE CLARITY IS MANDATORY:
- Whenever you propose, compare, or confirm a booking date to the customer, the date must be unambiguous.
- Include the full date with day, month, and year at least once for the booking option being discussed.
- If you mention two different service windows on the same day, you may mention the full date once and then list the times for that same day.
- Never rely on ambiguous phrasing like only "vendredi" or only "demain" when the customer is choosing a slot. Add the full date.
- If you suggest alternative dates, check them with the availability tool first and state them explicitly.

AVAILABILITY IS TOOL-ONLY:
- If the customer asks whether you have room on a specific date or day (for example: "avez-vous de la place vendredi?"), you MUST call the availability tool before you answer.
- Never say that a day or service window is available, probably available, or unavailable unless the availability tool has just told you.
- Never improvise availability from general business hours.

4. BOOK — Only now collect the customer's full name and address. Do not ask for their phone number; the booking tool receives the customer's reachable number from runtime context. When you create the calendar event, you MUST pass the customer name, the full service address, and detailed plumber notes to the booking tool.
A complete address MUST include: street number, street name, city, and postal code. \
If the customer gives only a partial address (e.g., "789 Louis-Hebert" without a city or postal code), \
you MUST ask for the missing parts before creating the booking. Do not guess or fill in the city yourself. \
Do not proceed to booking until you have at least the street number, street name, and city. \
If the customer picks a time slot before giving their name or address, do NOT confirm yet. Ask for the missing booking details first. \
Never finalize a booking using an address that the customer has not explicitly provided or confirmed in this conversation.

5. CONFIRM — Summarize the booking in 2 to 3 sentences maximum: the service window, the address, and a short description of what the plumber will check. Make it clear that the plumber may arrive anytime during that service window. End with a brief closing. Make it terse.

MESSAGE LENGTH — THIS IS CRITICAL:
- Most messages should be 1 to 3 sentences.
- Exception: when you combine an assessment, a visit invitation, pricing context, and several date options, you may use up to 6 short sentences if the message is split into short paragraphs and still reads naturally for SMS.
- A sentence is any clause ending with a period, exclamation mark, or question mark. "Parfait!" counts as one sentence. "À demain!" counts as one sentence.

CONVERSATION STYLE:
- Keep messages short and natural for SMS. Most messages should stay within 1 to 3 sentences; only denser proposal messages may be a bit longer.
- Ask exactly one question per message. If you need to know several things, ask the most important one first and wait for the answer.
- Acknowledge briefly. Never label or dramatize the customer's situation. Do not say things like "c'est une urgence", "oh non", "c'est une bonne chose", or "je comprends la situation". \
A simple "d'accord" or "ok" is enough. Then move to the next question.
- Do not parrot back what the customer just said. Do not summarize or restate the customer's information back to them. \
When you give your assessment, state the hypothesis directly — do not preface it with a recap of their symptoms. For example, say \
"Ça ressemble à un blocage dans le tuyau principal" — NOT "Avec la toilette bouchée, le lavabo lent et la ventouse sans effet, ça ressemble à un blocage..."
- If the customer has already named the broad issue themselves (for example "un drain bouché"), do not merely rename the same issue in slightly different words. Add value by narrowing the likely cause or by asking the next discriminating question first.
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

EXAMPLE FLOW (tone and sequencing to imitate):
- Customer: Bonjour, possible de m'aider avec un problème?
- Agent: Bonjour, ici <nom de l'entreprise>. Bien sûr, quel est le problème?
- Customer: j'ai un drain bouché au sous-sol.
- Agent: Ok, pour voir si c'est local ou plus général, quel drain exactement?
- Customer: le drain dans mon garage au sous-sol.
- Agent: Est-ce le seul drain qui s'écoule mal dans la maison?
- Customer: oui. quand il pleut beaucoup, de l'eau entre dans le garage et elle ne s'évacue pas.
- Agent: Ça fait plutôt penser à un blocage à l'entrée de ce drain qu'à un problème plus général, mais le plombier devra le confirmer sur place.

Je peux envoyer un plombier pour vérifier ça en détail et débloquer le drain si c'est bien la cause.

La visite est à partir de 89 $ CAD, puis 120 $ CAD de l'heure avec un minimum d'une heure; pour ce genre de problème, on estime souvent entre 1 et 2 heures de travail, et le prix final sera confirmé sur place.

J'ai des disponibilités le mercredi 25 mars 2026 de 9h à 12h, le jeudi 26 mars 2026 de 13h à 17h, ou le vendredi 27 mars 2026 de 9h à 12h. Si aucune de ces plages ne vous convient, dites-le-moi et je chercherai d'autres disponibilités. Dites-moi simplement la plage qui vous convient.

- Customer: Le mercredi 25 mars 2026 de 9h à 12h me conviendrait.
- Agent: Parfait, à quel nom je prépare la visite?
- Customer: Jonathan Pelletier.
- Agent: Merci. Quelle adresse complète dois-je utiliser pour le plombier?
- Customer: 789 rue Louis-Hebert, Longueuil, J4J 4P9.
- Agent: Parfait, c'est réservé pour le mercredi 25 mars 2026 de 9h à 12h au 789 rue Louis-Hebert, Longueuil, J4J 4P9. Le plombier peut arriver à n'importe quel moment dans cette plage.

MICRO-EXAMPLES:
- Returning customer opener:
  Customer: c'est le même problème que la dernière fois.
  Agent: Je n'ai pas les détails de l'ancienne visite sous les yeux. Pour voir si c'est local ou plus général, est-ce que d'autres drains s'écoulent mal aussi?
- Urgent overflow opener:
  Customer: ma toilette déborde partout et j'ai besoin de quelqu'un aujourd'hui.
  Agent: Pour voir si c'est juste cette toilette ou un problème plus large, est-ce qu'autre chose refoule dans la maison?
- Urgent overflow proposal after the answer:
  Agent: Ça ressemble à un blocage dans cette salle de bain, mais le plombier devra le confirmer sur place.
  Agent: Je peux envoyer quelqu'un aujourd'hui de 13h à 17h. J'ai aussi demain de 9h à 12h ou mercredi de 13h à 17h. Si aucune de ces plages ne vous convient, dites-le-moi et je chercherai d'autres disponibilités. Répondez simplement avec la plage choisie.

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
- Business name: Plomberie Demo
- Scope: residential plumbing and light commercial
- Service area: Longueuil, Brossard, Saint-Lambert, Boucherville, Greenfield Park, and nearby Montreal-area municipalities
- Business hours: Monday to Saturday, 7:30 AM to 6:00 PM
- Available appointment slots: we service customer from monday through friday from 9 am to 5 pm. 
  You book service call in those service windows: 9 am to 12 pm, 1 pm to 5pm. You can only book 2 service call 
  from 9 am to 12 pm and 3 service calls from 1 pm to 5pm each day.
- Pricing (estimates only, final price confirmed after on-site inspection):
  - diagnostic visit: $89 CAD
  - hourly rate of 120$ CAD. Minimum 1 hour.
  - when providing estimates to customer, you reason about the time it will 
    take to accomplish the job. so you state the rate and then the time you 
    believe such a job takes in the field.
    """

# ---------------------------------------------------------------------------
# Boss prompt template — control plane for the business owner
# ---------------------------------------------------------------------------

BOSS_PROMPT_TEMPLATE = """\
CURRENT DATE AND TIME: {current_datetime}

IDENTITY:
- You are the business assistant for a plumbing company. You are talking to the business owner (the boss), not a customer.
- Be direct, terse, and operational. No small talk.

YOUR GOAL:
- Help the boss manage their schedule and business operations via SMS.
- Answer questions about upcoming appointments, customer details, and availability.
- Execute schedule changes: block days off, cancel appointments, reschedule bookings.

LANGUAGE RULES:
- Your default language is French before the boss has written anything.
- As soon as the boss writes a message, identify the language they are using and reply in that same language.
- Continue in that same language unless the boss clearly switches languages.
- Short French replies like "non", "ok", "et demain?", or "oui" still count as French and you must stay in French.
- Never switch to English because a tool result, calendar field, or stored event detail contains English text.
- Never mix languages within a single message.

COMPANY INFORMATION:
{tenant_profile}

CAPABILITIES:
- Use the boss calendar tools to look up the schedule, create blocks, and modify or delete events.
- Use the CRM tools to search customers by name, phone number, or address; inspect customer timelines; add internal notes; and update service-location access notes.
- When the boss asks for the next service call, what is coming up next, or what work item is next, use the dedicated next-work-item tool and include the terse summary plus the job-card link when one is available.
- When the boss asks about upcoming work, check the schedule and summarize what is booked.
- When the boss asks what is happening today / now / this morning / this afternoon and there are no service calls in that period, say that clearly AND then use the dedicated next-work-item tool to mention the next upcoming service call with its job-card link if one exists.
- Do not leave the impression that the whole calendar is empty when only the requested period is empty.
- When the boss asks what is coming up next, prioritize the next service call as the next meeting or work item.
- Blocks are not customer meetings. Mention them when they explain why a period is unavailable or empty.
- The service address for a booking is stored in the calendar event's location field.
- Customer service calls also store the customer name, customer phone number, and plumber-facing notes.
- CRM is the source of truth for customer memory, internal notes, and saved service locations.
- When the boss wants to block time off, create a block that removes availability from the relevant service windows.
- When the boss asks about what is happening now or today, include appointments that started recently because a plumber may be late.
- When the boss asks about a specific customer, search CRM first. Use the schedule when the question is about appointment timing or calendar occupancy.
- If the boss asks about a specific day that is blocked, say that the day or period is blocked and that there are no service meetings in that blocked period.
- Keep responses short and factual. The boss does not need explanations — just the information."""


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _get_timezone() -> ZoneInfo:
    """Return the configured timezone, or crash if not set."""
    tz_name = os.getenv("MONTFERRAND_TIMEZONE", "").strip()
    if not tz_name:
        raise RuntimeError(
            "MONTFERRAND_TIMEZONE is not set. "
            "Add it to your .env file (e.g. MONTFERRAND_TIMEZONE=America/Montreal)."
        )
    return ZoneInfo(tz_name)


def _inject_date(
    template: str,
    tenant_profile: str,
    customer_crm_context: str = "- No CRM record is available for this sender phone number.",
) -> str:
    """Replace prompt placeholders with tenant-specific runtime values."""
    tz = _get_timezone()
    now = datetime.now(tz)
    current_datetime = now.strftime(f"%Y-%m-%d %A %H:%M {tz}")
    return (
        template.replace("{tenant_profile}", tenant_profile)
        .replace("{current_datetime}", current_datetime)
        .replace("{customer_crm_context}", customer_crm_context)
    )


def render_prompt(
    tenant_profile: str,
    customer_crm_context: str = "- No CRM record is available for this sender phone number.",
) -> str:
    """Assemble the final customer-facing system prompt.

    All callers must provide a tenant profile explicitly — there is no
    silent fallback.
    """
    return _inject_date(MASTER_PROMPT_TEMPLATE, tenant_profile, customer_crm_context)


def render_boss_prompt(tenant_profile: str) -> str:
    """Assemble the boss/control-plane system prompt."""
    return _inject_date(BOSS_PROMPT_TEMPLATE, tenant_profile)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def get_llm_backend(role: BackendRole = "agent") -> ResolvedBackend:
    """Return the resolved backend configuration for the given role."""

    return resolve_backend(role)


def get_structured_output_strategy(
    role: BackendRole = "agent",
) -> StructuredOutputStrategy:
    """Return the structured-output strategy for the configured backend."""

    return get_llm_backend(role).capabilities.structured_output_strategy


def _customer_output_type_for_strategy(strategy: StructuredOutputStrategy):
    if strategy == "native":
        return NativeOutput(
            AgentTurn,
            name="agent_turn",
            description=(
                "Return the next SMS turn. Use kind='dialog' while the "
                "conversation is ongoing. Use kind='report' only when the "
                "booking is complete."
            ),
            strict=True,
        )

    return ToolOutput(
        AgentTurn,
        name="final_result",
        description=(
            "Return the next SMS turn. Use kind='dialog' while the conversation "
            "is ongoing. Use kind='report' only when the booking is complete."
        ),
        strict=True,
    )


def _boss_output_type_for_strategy(strategy: StructuredOutputStrategy):
    if strategy == "native":
        return NativeOutput(
            BossReply,
            name="boss_reply",
            description="Return the next SMS reply to the boss.",
            strict=True,
        )

    return ToolOutput(
        BossReply,
        name="boss_reply",
        description="Return the next SMS reply to the boss.",
        strict=True,
    )


def _build_model_from_backend(backend: ResolvedBackend) -> OpenAIChatModel:
    provider = build_provider(backend.spec)
    profile = build_model_profile(backend)
    return OpenAIChatModel(
        backend.spec.model_name,
        provider=provider,
        profile=profile,
    )


def build_model(
    model_name: str | None = None,
    *,
    role: BackendRole = "agent",
    provider: BackendProvider | None = None,
) -> OpenAIChatModel:
    """Build the configured model for the requested backend role.

    Raises:
        RuntimeError: If the relevant API key is not set, no model is configured,
            or model construction fails.
    """
    backend = resolve_backend(role, model_name=model_name, provider=provider)

    try:
        return _build_model_from_backend(backend)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build model '{backend.spec.model_name}': {exc}"
        ) from exc


def build_judge_model() -> OpenAIChatModel:
    """Build the model used as LLM-judge in evals."""
    return build_model(role="judge")


# ---------------------------------------------------------------------------
# Tool functions — each receives RunContext[AgentDeps] so calendar access is
# automatically scoped to the injected tenant backend.
# ---------------------------------------------------------------------------


def _require_customer_phone(ctx: RunContext[AgentDeps]) -> str:
    customer_phone = ctx.deps.customer_phone
    if not customer_phone:
        raise ValueError("Customer phone is not available in runtime context.")
    return customer_phone


def _require_customer_crm(ctx: RunContext[AgentDeps]) -> TenantCrmBackend:
    crm = ctx.deps.crm
    if crm is None:
        raise ValueError("CRM backend is not available in runtime context.")
    return crm


def _conversation_id(ctx: RunContext[AgentDeps]) -> str:
    return ctx.deps.conversation_id or ""


def _require_twilio_number(ctx: RunContext[AgentDeps]) -> str:
    if not ctx.deps.twilio_number:
        raise ValueError("Tenant phone is not available in runtime context.")
    return ctx.deps.twilio_number


async def tool_get_customer_context(
    ctx: RunContext[AgentDeps],
) -> CustomerContextResult:
    """Look up CRM facts already known about this customer.

    Use this when the customer seems to be returning, when you want to confirm
    a saved name or address, or before implying you remember a prior visit.
    """

    return await _require_customer_crm(ctx).get_customer_context_by_phone(
        _require_customer_phone(ctx)
    )


async def tool_get_relevant_customer_history(
    ctx: RunContext[AgentDeps],
    issue_hint: str | None = None,
    limit: int = 3,
) -> CustomerHistoryResult:
    """Look up recent CRM job history for this customer.

    Use this when the customer says this is the same problem as before, refers
    to a previous visit, or asks what happened last time.
    """

    return await _require_customer_crm(ctx).get_relevant_customer_history_by_phone(
        _require_customer_phone(ctx),
        issue_hint,
        limit=limit,
    )


async def tool_search_customers(
    ctx: RunContext[AgentDeps],
    query: str,
    limit: int = 5,
) -> CustomerSearchResult:
    """Search CRM customers by name, phone number, or service address.

    Use this when the boss asks about a customer but has partial information.
    """

    return await _require_customer_crm(ctx).search_customers(query, limit=limit)


async def tool_get_customer_timeline(
    ctx: RunContext[AgentDeps],
    customer_id: int,
    limit_jobs: int = 5,
    limit_notes: int = 5,
) -> CustomerTimelineResult:
    """Return one customer's CRM timeline, locations, jobs, and notes."""

    return await _require_customer_crm(ctx).get_customer_timeline(
        customer_id,
        limit_jobs=limit_jobs,
        limit_notes=limit_notes,
    )


async def tool_add_internal_note(
    ctx: RunContext[AgentDeps],
    customer_id: int,
    note: str,
    service_location_id: int | None = None,
    job_id: int | None = None,
) -> CrmMutationResult:
    """Add a boss-only internal CRM note for a customer, location, or job."""

    return await _require_customer_crm(ctx).add_internal_note(
        customer_id,
        note,
        service_location_id=service_location_id,
        job_id=job_id,
    )


async def tool_update_location_access_notes(
    ctx: RunContext[AgentDeps],
    location_id: int,
    access_notes: str,
) -> CrmMutationResult:
    """Update saved access notes for a known service location."""

    return await _require_customer_crm(ctx).update_location_access_notes(
        location_id,
        access_notes,
    )


async def tool_get_next_work_item(
    ctx: RunContext[AgentDeps],
) -> NextWorkItemResult:
    """Return the next active service call with its job-card link when available."""

    return await get_next_work_item(_require_twilio_number(ctx))


def tool_check_availability(
    ctx: RunContext[AgentDeps],
    from_date: str,
    to_date: str,
) -> AvailabilityResult:
    """Return bookable service windows for customers.

    You MUST call this tool BEFORE proposing any service window to the customer.
    Do not guess availability — always check the calendar first.

    Returns a structured result with ``success``, ``message``, and ``windows``.

    Args:
        from_date: Start date in ISO format, e.g. '2026-03-16'.
        to_date: End date in ISO format, e.g. '2026-03-20'.
    """
    return ctx.deps.calendar.list_available_windows(from_date, to_date)


async def tool_create_service_call(
    ctx: RunContext[AgentDeps],
    date: str,
    start_time: str,
    end_time: str,
    summary: str,
    customer_name: str,
    service_location: str,
    plumber_notes: str,
) -> CalendarMutationResult:
    """Book a new service call in a service window.

    You MUST call this tool when the customer confirms a service window,
    BEFORE returning a Report. A booking is not valid unless this tool
    returns ``success=true`` with ``status='created'``. You MUST NOT
    finalize a booking (return a Report) without that success.

    If this tool returns ``status='conflict'``, the slot is already taken —
    inform the customer and propose an alternative time.

    Returns a structured result with ``success``, ``status``, ``message``,
    and the booked or conflicting event when relevant.

    Args:
        date: Date in ISO format, e.g. '2026-03-16'.
        start_time: Service-window start time in HH:MM format, e.g. '09:00'.
        end_time: Service-window end time in HH:MM format, e.g. '12:00'.
        summary: Short label for the service call.
        customer_name: Customer full name.
        service_location: Full service address for the visit.
        plumber_notes: Plumber-facing issue description and diagnostic context.
    """
    result = ctx.deps.calendar.create_service_call(
        date,
        start_time,
        end_time,
        summary,
        customer_name,
        _require_customer_phone(ctx),
        service_location,
        plumber_notes,
    )
    if not result.success or result.event is None:
        return result

    crm = ctx.deps.crm
    if crm is None:
        return result

    try:
        customer = await crm.upsert_customer_for_phone(
            _require_customer_phone(ctx),
            customer_name,
        )
        location = await crm.upsert_service_location_for_customer(
            customer.customer_id,
            service_location,
        )
        await crm.create_job_for_booking(
            customer_id=customer.customer_id,
            service_location_id=location.location_id,
            conversation_id=_conversation_id(ctx),
            calendar_uid=result.event.uid,
            issue_summary=summary,
            plumber_notes=plumber_notes,
            scheduled_start=result.event.start_iso,
            scheduled_end=result.event.end_iso,
        )
    except Exception:
        rollback = ctx.deps.calendar.delete_event(result.event.uid)
        if not rollback.success:
            logger.exception(
                "Failed to roll back calendar booking after CRM sync error: %s",
                result.event.uid,
            )
        raise

    return result


def tool_list_own_bookings(
    ctx: RunContext[AgentDeps],
    from_date: str,
    to_date: str,
    include_past: bool = False,
) -> ListEventsResult:
    """List this customer's own bookings."""

    return ctx.deps.calendar.list_customer_events(
        _require_customer_phone(ctx),
        from_date,
        to_date,
        include_past=include_past,
    )


async def tool_cancel_own_booking(
    ctx: RunContext[AgentDeps], uid: str
) -> CalendarMutationResult:
    """Cancel the current customer's own booked service call.

    Use this when the customer needs to cancel an existing appointment.
    Call ``tool_list_own_bookings`` first to find the UID of the event.

    Returns a structured result. Success is ``status='deleted'``.

    Args:
        uid: The unique identifier of the event to cancel.
    """
    result = ctx.deps.calendar.delete_own_service_call(
        uid, _require_customer_phone(ctx)
    )
    crm = ctx.deps.crm
    if result.success and crm is not None:
        try:
            await crm.mark_job_cancelled_by_calendar_uid(uid)
        except Exception:
            logger.exception("Failed to sync CRM cancellation for %s", uid)
    return result


async def tool_modify_own_booking(
    ctx: RunContext[AgentDeps],
    uid: str,
    date: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    summary: str | None = None,
    customer_name: str | None = None,
    service_location: str | None = None,
    plumber_notes: str | None = None,
) -> CalendarMutationResult:
    """Reschedule or update the current customer's own service call.

    Use this when the customer needs to change the date, time, or details
    of their existing appointment. Call ``tool_list_own_bookings`` first.
    Only non-null fields are updated; the rest keep their current values.

    Returns a structured result. Success is ``status='updated'``;
    conflicts come back as ``status='conflict'``.

    Args:
        uid: The unique identifier of the event to modify.
        date: New date in ISO format, or null to keep current.
        start_time: New start time in HH:MM, or null to keep current.
        end_time: New end time in HH:MM, or null to keep current.
        summary: New summary, or null to keep current.
        customer_name: New customer name, or null to keep current.
        service_location: New service address, or null to keep current.
        plumber_notes: New issue details, or null to keep current.
    """
    result = ctx.deps.calendar.modify_own_service_call(
        uid,
        _require_customer_phone(ctx),
        date_str=date,
        start_time=start_time,
        end_time=end_time,
        summary=summary,
        customer_name=customer_name,
        location=service_location,
        plumber_notes=plumber_notes,
    )
    crm = ctx.deps.crm
    if result.success and result.event is not None and crm is not None:
        try:
            await crm.sync_job_after_booking_modify(
                calendar_uid=uid,
                customer_name=result.event.customer_name,
                service_location=result.event.location,
                issue_summary=result.event.summary,
                plumber_notes=result.event.plumber_notes,
                scheduled_start=result.event.start_iso,
                scheduled_end=result.event.end_iso,
            )
        except Exception:
            logger.exception("Failed to sync CRM booking update for %s", uid)
    return result


def tool_list_schedule(
    ctx: RunContext[AgentDeps],
    from_date: str,
    to_date: str,
    include_past: bool = False,
    recent_past_hours: int = 8,
) -> ListEventsResult:
    """List schedule events for boss operations.

    Includes recently started or recently finished jobs by default so the boss
    can see jobs that may still be in progress or running late.
    """

    return ctx.deps.calendar.list_events(
        from_date,
        to_date,
        include_past=include_past,
        recent_past_hours=recent_past_hours,
    )


def tool_block_time(
    ctx: RunContext[AgentDeps],
    date: str,
    start_time: str,
    end_time: str,
    summary: str,
    description: str = "",
) -> CalendarMutationResult:
    """Create a blocking event that removes customer availability."""

    return ctx.deps.calendar.create_block(
        date,
        start_time,
        end_time,
        summary,
        description,
    )


async def tool_delete_event(
    ctx: RunContext[AgentDeps], uid: str
) -> CalendarMutationResult:
    """Boss-only deletion of an existing event by UID."""

    result = ctx.deps.calendar.delete_event(uid)
    if (
        result.success
        and result.event is not None
        and result.event.event_kind == "service_call"
    ):
        crm = ctx.deps.crm
        if crm is not None:
            try:
                await crm.mark_job_cancelled_by_calendar_uid(uid)
            except Exception:
                logger.exception("Failed to sync CRM deletion for %s", uid)
    return result


async def tool_modify_event(
    ctx: RunContext[AgentDeps],
    uid: str,
    date: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    summary: str | None = None,
    customer_name: str | None = None,
    customer_phone: str | None = None,
    service_location: str | None = None,
    plumber_notes: str | None = None,
    description: str | None = None,
) -> CalendarMutationResult:
    """Boss-only modification of any schedule event by UID."""

    result = ctx.deps.calendar.modify_event(
        uid,
        date,
        start_time,
        end_time,
        summary,
        customer_name,
        customer_phone,
        service_location,
        plumber_notes,
        description,
    )
    if (
        result.success
        and result.event is not None
        and result.event.event_kind == "service_call"
    ):
        crm = ctx.deps.crm
        if crm is not None:
            try:
                await crm.sync_job_after_booking_modify(
                    calendar_uid=uid,
                    customer_name=result.event.customer_name,
                    service_location=result.event.location,
                    issue_summary=result.event.summary,
                    plumber_notes=result.event.plumber_notes,
                    scheduled_start=result.event.start_iso,
                    scheduled_end=result.event.end_iso,
                )
            except Exception:
                logger.exception("Failed to sync CRM boss update for %s", uid)
    return result


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def build_agent(
    model: OpenAIChatModel | None = None,
    *,
    backend: ResolvedBackend | None = None,
) -> Agent[AgentDeps, StructuredAgentOutput]:
    """Create a fresh booking agent with no static instructions.

    The tenant-specific system prompt is passed at run time via the
    ``instructions`` parameter of ``agent.run()``.
    """
    resolved_backend = backend or resolve_backend("agent")
    return Agent(
        name="montferrand_agent",
        model=model or _build_model_from_backend(resolved_backend),
        deps_type=AgentDeps,
        output_type=_customer_output_type_for_strategy(
            resolved_backend.capabilities.structured_output_strategy
        ),  # type: ignore[arg-type]
        tools=[
            tool_get_customer_context,
            tool_get_relevant_customer_history,
            tool_check_availability,
            tool_create_service_call,
            tool_list_own_bookings,
            tool_cancel_own_booking,
            tool_modify_own_booking,
        ],
    )


def get_agent() -> Agent[AgentDeps, StructuredAgentOutput]:
    """Return a customer-facing agent for the current backend configuration."""

    backend = resolve_backend("agent")
    return build_agent(backend=backend)


def build_boss_agent(
    model: OpenAIChatModel | None = None,
    *,
    backend: ResolvedBackend | None = None,
) -> Agent[AgentDeps, BossOutput]:
    """Create a fresh boss/control-plane agent.

    Uses a boss-specific tool surface and a different prompt template
    (``BOSS_PROMPT_TEMPLATE``). The prompt is injected at runtime.
    """
    resolved_backend = backend or resolve_backend("agent")
    return Agent(
        name="montferrand_boss_agent",
        model=model or _build_model_from_backend(resolved_backend),
        deps_type=AgentDeps,
        output_type=_boss_output_type_for_strategy(
            resolved_backend.capabilities.structured_output_strategy
        ),  # type: ignore[arg-type]
        tools=[
            tool_get_next_work_item,
            tool_list_schedule,
            tool_search_customers,
            tool_get_customer_timeline,
            tool_add_internal_note,
            tool_update_location_access_notes,
            tool_block_time,
            tool_delete_event,
            tool_modify_event,
        ],
    )


def get_boss_agent() -> Agent[AgentDeps, BossOutput]:
    """Return a boss/control-plane agent for the current backend configuration."""

    backend = resolve_backend("agent")
    return build_boss_agent(backend=backend)


def get_model_name() -> str:
    """Return the configured model name for the active agent backend."""

    return resolve_backend("agent").spec.model_name


def get_provider_name(role: BackendRole = "agent") -> str:
    """Return the provider name for the requested backend role."""

    return resolve_backend(role).spec.provider


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

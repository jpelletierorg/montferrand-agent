"""Simulated customer agent used exclusively by the eval harness.

This module defines a simple pydantic-ai Agent that role-plays as a customer
sending SMS messages to a plumbing company.  Each eval scenario provides a
persona describing who the customer is, what their problem is, and the
details the customer should reveal when asked (name, address, etc.).

The customer agent is stateless — conversation continuity comes from passing
message_history on each call, just like the booking agent.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from montferrand_agent.agent import build_judge_model


class CustomerAgentError(RuntimeError):
    """Raised when the simulated customer agent fails."""


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_CUSTOMER_SYSTEM_PROMPT = """\
You are a customer sending SMS messages to a plumbing company's after-hours \
booking service.

SCENARIO:
{persona}

BEHAVIOR:
- You are reaching out because you have a plumbing problem.
- Let the plumber's service guide the conversation. Answer their questions \
when asked.
- Stay consistent with the scenario details above. Use the name, address, \
and problem description provided.
- Write short, natural SMS messages — 1 to 2 sentences, like a real person \
texting.
- If the plumber asks for information that is in your scenario, provide it.
- If the plumber proposes an appointment slot, accept one that works for you \
based on your scenario.
- Do not volunteer all your information at once. Provide details as the \
plumber asks for them, like a real conversation.
- Write in the language specified in your scenario (French by default)."""

_OPENING_PROMPT = "Send your first SMS to the plumbing company about your problem."

# ---------------------------------------------------------------------------
# Customer agent construction
# ---------------------------------------------------------------------------


def build_customer_agent(persona: str) -> Agent[None, str]:
    """Create a customer agent for a given eval scenario persona."""
    return Agent(
        name="eval_customer",
        model=build_judge_model(),
        output_type=str,
        system_prompt=_CUSTOMER_SYSTEM_PROMPT.format(persona=persona),
    )


async def customer_reply(
    agent: Agent[None, str],
    message: str | None = None,
    history: list[ModelMessage] | None = None,
) -> tuple[str, list[ModelMessage]]:
    """Generate the next customer message.

    When *message* is None (or omitted), generates the customer's opening
    message to kick off the conversation.  Otherwise, replies to the agent's
    latest message using the provided conversation *history*.

    Returns the customer's message text and the updated message history.

    Raises:
        CustomerAgentError: If the customer agent call fails.
    """
    prompt = message or _OPENING_PROMPT

    try:
        if history is None:
            result = await agent.run(prompt)
        else:
            result = await agent.run(prompt, message_history=history)
    except Exception as exc:
        raise CustomerAgentError(f"Customer agent call failed: {exc}") from exc

    return result.output, result.all_messages()

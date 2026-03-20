"""Output types for the Montferrand booking agent.

The app exposes two public result types:
- Dialog: the conversation continues, the agent needs more information.
- Report: the agent has everything it needs to book a service visit.

Internally, the LLM returns a single structured ``AgentTurn`` object. Using a
single schema keeps native structured-output backends like Inception and Claude
via OpenRouter on a simpler, more stable protocol than a top-level union.

Field descriptions and examples are passed into the LLM context window via the
JSON schema, so they must clearly describe what each field is for and how the
model should fill it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Dialog(BaseModel):
    """The conversation is still in progress. Return this when you do not yet
    have ALL of the following: the customer's full name, their street address,
    a clear description of the plumbing problem, and a confirmed appointment
    window. Keep asking questions until you have every piece."""

    message: str = Field(
        description=(
            "The SMS message to send back to the customer. Keep it short and "
            "natural for SMS: 1 to 3 sentences maximum. Ask only one "
            "clarifying question at a time. Write in the same language the "
            "customer is using."
        ),
        examples=[
            "Bonjour. On peut vous aider avec cette situation. Est-ce que l'eau coule activement en ce moment?",
            "Merci. Pouvez-vous me donner votre adresse pour qu'on planifie la visite?",
            "OK, I can help with that. Is the leak actively dripping right now?",
        ],
    )


class BossReply(BaseModel):
    """Open-ended boss/control-plane reply.

    Boss conversations never terminate in a booking report state; each turn is
    simply another operational reply.
    """

    message: str = Field(
        description=(
            "The SMS message to send back to the boss. Keep it direct, concise, "
            "and operational."
        )
    )


class Report(BaseModel):
    """All required information has been collected AND the customer has
    confirmed the appointment. Return this only when you have the customer's
    name, address, a clear issue description, and a mutually agreed time slot.
    Once you return a Report the conversation is over."""

    message: str = Field(
        description=(
            "Final confirmation SMS sent to the customer. Summarize the "
            "booking: repeat the date and time slot, the address, and a "
            "one-line description of the problem so the customer can verify "
            "everything is correct."
        ),
        examples=[
            "Parfait Marie-Claude. On vous envoie un plombier demain entre 8h et 10h au 123 rue des Erables a Longueuil pour la fuite sous votre evier. A demain.",
            "All set John. A plumber will be at 45 Oak Street in Brossard tomorrow between 1 PM and 3 PM to look at your clogged drain. See you then.",
        ],
    )

    customer_name: str = Field(
        description=(
            "The customer's full name exactly as they provided it during the "
            "conversation. Do not invent or guess a name."
        ),
        examples=["Marie-Claude Tremblay", "John Smith"],
    )

    service_location: str = Field(
        description=(
            "The street address where the plumber must go. House number, street name, city and postal code."
        ),
        examples=[
            "123 rue des Erables, Longueuil",
            "45 Oak Street, Brossard, J4W 2T5",
        ],
    )

    issue_description: str = Field(
        description=(
            "A clear description of the plumbing problem for the plumber doing "
            "the on-site visit. Include the customer's reported symptoms, any "
            "diagnostic observations from your questions or photos, and your "
            "assessment of what the likely issue is. Two to four sentences."
        ),
        examples=[
            "Fuite active sous l'evier de cuisine, probablement au niveau du joint du siphon. Le client rapporte un ecoulement lent mais constant depuis ce matin. L'eau s'accumule dans l'armoire sous l'evier.",
            "Toilette du rez-de-chaussee completement bouchee, l'eau deborde sur le plancher. La ventouse n'a pas fonctionne. Possiblement un blocage dans le renvoi principal vu que le client mentionne un ralentissement dans les autres drains aussi.",
        ],
    )

    appointment_window: str = Field(
        description=(
            "The agreed-upon date and time slot for the visit, for example "
            "'demain 8h a 10h' or 'aujourd'hui 15h a 17h'. Must reflect what "
            "the customer actually confirmed, not just what was proposed."
        ),
        examples=[
            "demain 8h a 10h",
            "aujourd'hui 15h a 17h",
            "demain 13h a 15h",
        ],
    )


class AgentTurn(BaseModel):
    """Single structured result returned by the LLM on each turn.

    ``kind='dialog'`` means the conversation continues and only ``message`` is
    required. ``kind='report'`` means the booking is complete and all report
    fields must be present.
    """

    kind: Literal["dialog", "report"] = Field(
        description=(
            "What kind of turn this is. Use 'dialog' when the conversation is "
            "still in progress. Use 'report' only when the booking is complete."
        )
    )

    message: str = Field(
        description=(
            "The SMS message to send to the user right now. Keep it short and "
            "natural for SMS."
        )
    )

    customer_name: str | None = Field(
        default=None,
        description=(
            "The customer's full name when kind='report'. Leave null while the "
            "conversation is still in progress."
        ),
    )
    service_location: str | None = Field(
        default=None,
        description=(
            "The street address for the visit when kind='report'. Leave null "
            "while the conversation is still in progress."
        ),
    )
    issue_description: str | None = Field(
        default=None,
        description=(
            "A concise plumber-facing description of the problem when "
            "kind='report'. Leave null while the conversation is still in "
            "progress."
        ),
    )
    appointment_window: str | None = Field(
        default=None,
        description=(
            "The mutually agreed appointment slot when kind='report'. Leave "
            "null while the conversation is still in progress."
        ),
    )

    @model_validator(mode="after")
    def _require_report_fields(self) -> AgentTurn:
        if self.kind == "report":
            missing = [
                field_name
                for field_name in (
                    "customer_name",
                    "service_location",
                    "issue_description",
                    "appointment_window",
                )
                if not getattr(self, field_name)
            ]
            if missing:
                names = ", ".join(missing)
                raise ValueError(
                    f"Report turns must include all booking fields. Missing: {names}."
                )
        return self

    def to_public_result(self) -> Dialog | Report:
        """Convert the internal turn schema to the app's public result types."""

        if self.kind == "dialog":
            return Dialog(message=self.message)

        return Report(
            message=self.message,
            customer_name=self.customer_name or "",
            service_location=self.service_location or "",
            issue_description=self.issue_description or "",
            appointment_window=self.appointment_window or "",
        )

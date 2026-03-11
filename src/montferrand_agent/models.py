"""Output types for the Montferrand booking agent.

The agent returns exactly one of two types on each turn:
- Dialog: the conversation continues, the agent needs more information.
- Report: the agent has everything it needs to book a service visit.

Field descriptions and examples are passed into the LLM context window via
the JSON schema, so they must clearly describe what each field is for and
how the model should fill it.
"""

from pydantic import BaseModel, Field


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

from __future__ import annotations

import itertools
import os
import uuid

from locust import HttpUser, between, task

_CUSTOMER_COUNTER = itertools.count(1)


def _tenant_phone() -> str:
    return os.getenv("MONTFERRAND_LOAD_TENANT_PHONE", "+15551234567")


def _customer_prefix() -> str:
    return os.getenv("MONTFERRAND_LOAD_CUSTOMER_PREFIX", "+1438555")


def _new_customer_number() -> str:
    return f"{_customer_prefix()}{next(_CUSTOMER_COUNTER):04d}"


class MontferrandSmsUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self) -> None:
        self.customer_number = _new_customer_number()

    def _post_sms(self, body: str) -> None:
        self.client.post(
            "/sms",
            data={
                "To": _tenant_phone(),
                "From": self.customer_number,
                "Body": body,
                "MessageSid": f"SM{uuid.uuid4().hex[:20]}",
            },
            name="POST /sms",
        )

    @task(3)
    def first_contact(self) -> None:
        self._post_sms("bonjour j'ai un drain de plancher bloque")

    @task(2)
    def returning_customer(self) -> None:
        self._post_sms("c'est le meme probleme que la derniere fois")

    @task(1)
    def booking_followup(self) -> None:
        self._post_sms("lundi matin ca marche pour moi")

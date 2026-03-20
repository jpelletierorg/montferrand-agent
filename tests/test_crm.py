import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

from montferrand_agent.agent import (
    AgentDeps,
    tool_cancel_own_booking,
    tool_create_service_call,
    tool_get_customer_context,
    tool_get_relevant_customer_history,
    tool_modify_own_booking,
)
from montferrand_agent.calendar import get_tenant_calendar
from montferrand_agent.crm import (
    TenantCrmMissingError,
    crm_migrations_dir,
    get_tenant_crm,
    migrate_all_tenant_crm,
    migrate_tenant_crm,
    provision_tenant_crm,
    tenant_crm_db_path,
    tenant_crm_dir,
    verify_tenant_crm,
)
from montferrand_agent.tenant import phone_to_filename, save_tenant_profile

from .conftest import CUSTOMER_NUMBER, TENANT_PHONE, TEST_PROFILE, TWILIO_NUMBER


class TestCrmPaths:
    def test_tenant_crm_dir_uses_tenant_hash(self, isolated_data_dir):
        assert tenant_crm_dir(TENANT_PHONE) == (
            isolated_data_dir / "tenant" / "db" / phone_to_filename(TENANT_PHONE)
        )

    def test_tenant_crm_db_path_points_to_crm_sqlite(self, isolated_data_dir):
        assert tenant_crm_db_path(TENANT_PHONE) == (
            isolated_data_dir
            / "tenant"
            / "db"
            / phone_to_filename(TENANT_PHONE)
            / "crm.sqlite3"
        )


class TestCrmProvisioning:
    def test_provision_creates_db_and_schema(self, isolated_data_dir, fake_dbmate):
        path = provision_tenant_crm(TENANT_PHONE)

        assert path.exists()
        conn = sqlite3.connect(path)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            assert "crm_customers" in tables
            assert "customer_phones" in tables
            assert "job_tokens" in tables
            versions = {
                row[0] for row in conn.execute("SELECT version FROM schema_migrations")
            }
            expected_versions = {
                path.name.split("_", maxsplit=1)[0]
                for path in crm_migrations_dir().glob("*.sql")
            }
            assert versions == expected_versions
        finally:
            conn.close()

    def test_migrate_missing_db_raises(self, isolated_data_dir):
        with pytest.raises(TenantCrmMissingError, match="CRM DB is missing"):
            migrate_tenant_crm(TENANT_PHONE)

    def test_verify_missing_db_raises(self, isolated_data_dir):
        with pytest.raises(TenantCrmMissingError, match="CRM DB is missing"):
            verify_tenant_crm(TENANT_PHONE)

    def test_migrate_all_requires_existing_db(
        self, isolated_tenant_dir, isolated_data_dir
    ):
        save_tenant_profile(TENANT_PHONE, TEST_PROFILE)

        with pytest.raises(TenantCrmMissingError, match="CRM DB is missing"):
            migrate_all_tenant_crm()


@pytest.mark.asyncio
async def test_get_tenant_crm_lookup_customer_by_phone(isolated_data_dir, fake_dbmate):
    path = provision_tenant_crm(TENANT_PHONE)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT INTO crm_customers(id, display_name, preferred_language, created_at, updated_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                1,
                "Jonathan Pelletier",
                "fr",
                "2026-03-20T10:00:00",
                "2026-03-20T10:00:00",
                "2026-03-20T10:00:00",
            ),
        )
        conn.execute(
            "INSERT INTO customer_phones(customer_id, phone_e164, is_primary, created_at) VALUES (?, ?, ?, ?)",
            (1, "+14381112222", 1, "2026-03-20T10:00:00"),
        )
        conn.commit()
    finally:
        conn.close()

    backend = get_tenant_crm(TENANT_PHONE)
    customer = await backend.lookup_customer_by_phone("+14381112222")

    assert customer is not None
    assert customer.customer_id == 1
    assert customer.display_name == "Jonathan Pelletier"
    assert customer.preferred_language == "fr"


@pytest.mark.asyncio
async def test_tool_get_customer_context_returns_known_customer(
    isolated_data_dir, fake_dbmate
):
    provision_tenant_crm(TWILIO_NUMBER)
    crm = get_tenant_crm(TWILIO_NUMBER)
    customer = await crm.upsert_customer_for_phone(
        CUSTOMER_NUMBER, "Jonathan Pelletier"
    )
    await crm.upsert_service_location_for_customer(
        customer.customer_id,
        "123 rue Test, Longueuil, J4K 1A1",
    )
    ctx = cast(
        Any,
        SimpleNamespace(
            deps=AgentDeps(
                calendar=get_tenant_calendar(TWILIO_NUMBER),
                crm=crm,
                customer_phone=CUSTOMER_NUMBER,
                conversation_id="conv1",
            )
        ),
    )

    result = await tool_get_customer_context(ctx)

    assert result.known_customer is True
    assert result.customer_name == "Jonathan Pelletier"
    assert result.primary_location == "123 rue Test, Longueuil, J4K 1A1"


@pytest.mark.asyncio
async def test_tool_get_relevant_customer_history_returns_recent_jobs(
    isolated_data_dir, fake_dbmate
):
    provision_tenant_crm(TWILIO_NUMBER)
    crm = get_tenant_crm(TWILIO_NUMBER)
    customer = await crm.upsert_customer_for_phone(
        CUSTOMER_NUMBER, "Jonathan Pelletier"
    )
    location = await crm.upsert_service_location_for_customer(
        customer.customer_id,
        "123 rue Test, Longueuil, J4K 1A1",
    )
    await crm.create_job_for_booking(
        customer_id=customer.customer_id,
        service_location_id=location.location_id,
        conversation_id="conv1",
        calendar_uid="uid-1",
        issue_summary="Drain sous-sol bloque",
        plumber_notes="Drain sous-sol bloque depuis hier.",
        scheduled_start="2030-03-20T09:00:00-04:00",
        scheduled_end="2030-03-20T12:00:00-04:00",
    )
    ctx = cast(
        Any,
        SimpleNamespace(
            deps=AgentDeps(
                calendar=get_tenant_calendar(TWILIO_NUMBER),
                crm=crm,
                customer_phone=CUSTOMER_NUMBER,
                conversation_id="conv1",
            )
        ),
    )

    result = await tool_get_relevant_customer_history(ctx, issue_hint="sous-sol")

    assert result.success is True
    assert result.customer_name == "Jonathan Pelletier"
    assert len(result.jobs) == 1
    assert result.jobs[0].issue_summary == "Drain sous-sol bloque"


@pytest.mark.asyncio
async def test_booking_tools_sync_crm_customer_and_job(isolated_data_dir, fake_dbmate):
    provision_tenant_crm(TWILIO_NUMBER)
    crm = get_tenant_crm(TWILIO_NUMBER)
    calendar = get_tenant_calendar(TWILIO_NUMBER)
    booking_date = (date.today() + timedelta(days=2)).isoformat()
    ctx = cast(
        Any,
        SimpleNamespace(
            deps=AgentDeps(
                calendar=calendar,
                crm=crm,
                customer_phone=CUSTOMER_NUMBER,
                conversation_id="conv-booking",
            )
        ),
    )

    created = await tool_create_service_call(
        ctx,
        booking_date,
        "09:00",
        "12:00",
        "Drain sous-sol bloque",
        "Jonathan Pelletier",
        "123 rue Test, Longueuil, J4K 1A1",
        "Drain sous-sol bloque depuis hier.",
    )

    assert created.success is True
    assert created.event is not None
    context = await crm.get_customer_context_by_phone(CUSTOMER_NUMBER)
    assert context.known_customer is True
    assert context.customer_name == "Jonathan Pelletier"
    assert context.primary_location == "123 rue Test, Longueuil, J4K 1A1"
    assert len(context.recent_jobs) == 1
    assert context.recent_jobs[0].calendar_uid == created.event.uid

    updated = await tool_modify_own_booking(
        ctx,
        created.event.uid,
        summary="Drain garage encore bloque",
        service_location="456 rue Nouvelle, Brossard, J4Z 1A1",
        plumber_notes="Nouvelle adresse, meme probleme de drain.",
    )

    assert updated.success is True
    assert updated.event is not None
    updated_context = await crm.get_customer_context_by_phone(CUSTOMER_NUMBER)
    assert updated_context.primary_location == "456 rue Nouvelle, Brossard, J4Z 1A1"
    assert updated_context.recent_jobs[0].issue_summary == "Drain garage encore bloque"

    cancelled = await tool_cancel_own_booking(ctx, created.event.uid)

    assert cancelled.success is True
    cancelled_history = await crm.get_relevant_customer_history_by_phone(
        CUSTOMER_NUMBER
    )
    assert cancelled_history.jobs[0].status == "cancelled"


@pytest.mark.asyncio
async def test_job_token_and_next_job_lookup(isolated_data_dir, fake_dbmate):
    provision_tenant_crm(TWILIO_NUMBER)
    crm = get_tenant_crm(TWILIO_NUMBER)
    customer = await crm.upsert_customer_for_phone(
        CUSTOMER_NUMBER, "Jonathan Pelletier"
    )
    location = await crm.upsert_service_location_for_customer(
        customer.customer_id,
        "789 Louis-Hebert, Longueuil, J4K 1A1",
    )
    job_id = await crm.create_job_for_booking(
        customer_id=customer.customer_id,
        service_location_id=location.location_id,
        conversation_id="conv-token",
        calendar_uid="uid-token",
        issue_summary="Drain de plancher bloque",
        plumber_notes="Inspecter le drain du sous-sol.",
        scheduled_start="2030-03-24T09:00:00-04:00",
        scheduled_end="2030-03-24T12:00:00-04:00",
    )

    token = await crm.issue_job_token(job_id, ttl_hours=24)
    card = await crm.get_job_card_by_token(token)
    next_job = await crm.get_next_open_job()

    assert card is not None
    assert card.success is True
    assert card.job_id == job_id
    assert next_job is not None
    assert next_job.success is True
    assert next_job.job_id == job_id

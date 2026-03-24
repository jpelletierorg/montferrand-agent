import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

import montferrand_agent.crm as crm_module

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
    CrmMutationResult,
    CustomerSearchResult,
    ensure_existing_tenant_crm,
    ensure_tenant_crm,
    TenantCrmMissingError,
    crm_migrations_dir,
    get_tenant_crm,
    migrate_all_tenant_crm,
    migrate_existing_tenant_crm,
    migrate_tenant_crm,
    provision_tenant_crm,
    render_customer_context_for_prompt,
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

    def test_crm_migrations_dir_finds_packaged_app_root(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        app_root = tmp_path / "app"
        migrations = app_root / "db" / "crm" / "migrations"
        migrations.mkdir(parents=True)
        fake_module = (
            app_root
            / ".venv"
            / "lib"
            / "python3.13"
            / "site-packages"
            / "montferrand_agent"
            / "crm.py"
        )
        fake_module.parent.mkdir(parents=True)
        fake_module.write_text("", encoding="utf-8")

        monkeypatch.setattr(crm_module, "__file__", str(fake_module))

        assert crm_migrations_dir() == migrations


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

    def test_migrate_existing_skips_missing_db(
        self, isolated_tenant_dir, isolated_data_dir, fake_dbmate
    ):
        save_tenant_profile(TENANT_PHONE, TEST_PROFILE)
        save_tenant_profile(TWILIO_NUMBER, TEST_PROFILE)
        expected = provision_tenant_crm(TWILIO_NUMBER)

        migrated, missing = migrate_existing_tenant_crm()

        assert migrated == [expected]
        assert missing == [TENANT_PHONE]

    def test_ensure_tenant_crm_provisions_missing_db(
        self, isolated_tenant_dir, isolated_data_dir, fake_dbmate
    ):
        save_tenant_profile(TENANT_PHONE, TEST_PROFILE)

        ensured = ensure_tenant_crm(TENANT_PHONE)

        assert ensured == tenant_crm_db_path(TENANT_PHONE)
        assert ensured.exists()

    def test_ensure_existing_tenant_crm_migrates_and_provisions(
        self, isolated_tenant_dir, isolated_data_dir, fake_dbmate
    ):
        save_tenant_profile(TENANT_PHONE, TEST_PROFILE)
        save_tenant_profile(TWILIO_NUMBER, TEST_PROFILE)
        expected_migrated = provision_tenant_crm(TWILIO_NUMBER)

        migrated, provisioned = ensure_existing_tenant_crm()

        assert migrated == [expected_migrated]
        assert provisioned == [TENANT_PHONE]
        assert tenant_crm_db_path(TENANT_PHONE).exists()


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
async def test_render_customer_context_prefers_active_jobs(
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
        conversation_id="conv-active",
        calendar_uid="uid-active",
        issue_summary="Drain sous-sol bloque",
        plumber_notes="Drain sous-sol bloque depuis hier.",
        scheduled_start="2030-03-20T09:00:00-04:00",
        scheduled_end="2030-03-20T12:00:00-04:00",
    )

    context = await crm.get_customer_context_by_phone(CUSTOMER_NUMBER)
    prompt_context = render_customer_context_for_prompt(context)

    assert context.active_jobs[0].issue_summary == "Drain sous-sol bloque"
    assert "Known customer name: Jonathan Pelletier" in prompt_context
    assert (
        "One saved service address: 123 rue Test, Longueuil, J4K 1A1" in prompt_context
    )
    assert "Active jobs/bookings:" in prompt_context
    assert "Short past history:" not in prompt_context


@pytest.mark.asyncio
async def test_render_customer_context_falls_back_to_short_history(
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
    job_id = await crm.create_job_for_booking(
        customer_id=customer.customer_id,
        service_location_id=location.location_id,
        conversation_id="conv-history",
        calendar_uid="uid-history",
        issue_summary="Fuite sous evier",
        plumber_notes="Ancienne fuite sous evier.",
        scheduled_start="2030-03-20T09:00:00-04:00",
        scheduled_end="2030-03-20T12:00:00-04:00",
    )
    await crm.update_job_status(job_id, "completed")

    context = await crm.get_customer_context_by_phone(CUSTOMER_NUMBER)
    prompt_context = render_customer_context_for_prompt(context)

    assert context.active_jobs == []
    assert len(context.recent_jobs) == 1
    assert context.recent_jobs[0].issue_summary == "Fuite sous evier"
    assert "Active jobs/bookings:" not in prompt_context
    assert "Short past history:" in prompt_context


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
    assert len(context.active_jobs) == 1
    assert context.active_jobs[0].calendar_uid == created.event.uid
    assert context.recent_jobs == []

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
    assert updated_context.active_jobs[0].issue_summary == "Drain garage encore bloque"

    cancelled = await tool_cancel_own_booking(ctx, created.event.uid)

    assert cancelled.success is True
    cancelled_history = await crm.get_relevant_customer_history_by_phone(
        CUSTOMER_NUMBER
    )
    assert cancelled_history.jobs[0].status == "cancelled"


@pytest.mark.asyncio
async def test_search_customers_and_timeline_support_boss_lookup(
    isolated_data_dir, fake_dbmate
):
    provision_tenant_crm(TWILIO_NUMBER)
    crm = get_tenant_crm(TWILIO_NUMBER)
    customer = await crm.upsert_customer_for_phone(
        CUSTOMER_NUMBER, "Jonathan Pelletier"
    )
    location = await crm.upsert_service_location_for_customer(
        customer.customer_id,
        "789 Louis-Hebert, Longueuil, J4K 1A1",
    )
    await crm.create_job_for_booking(
        customer_id=customer.customer_id,
        service_location_id=location.location_id,
        conversation_id="conv-search",
        calendar_uid="uid-search",
        issue_summary="Drain garage bloque",
        plumber_notes="Drain bloque quand il pleut.",
        scheduled_start="2030-03-21T13:00:00-04:00",
        scheduled_end="2030-03-21T17:00:00-04:00",
    )
    await crm.add_internal_note(
        customer.customer_id,
        "Client prefere le texto.",
        service_location_id=location.location_id,
    )

    search = await crm.search_customers("Pelletier")

    assert search.success is True
    assert len(search.matches) == 1
    assert search.matches[0].customer_name == "Jonathan Pelletier"
    assert search.matches[0].primary_location == "789 Louis-Hebert, Longueuil, J4K 1A1"

    timeline = await crm.get_customer_timeline(customer.customer_id)

    assert timeline.success is True
    assert timeline.customer_name == "Jonathan Pelletier"
    assert CUSTOMER_NUMBER in timeline.phones
    assert (
        timeline.known_locations[0].formatted_address
        == "789 Louis-Hebert, Longueuil, J4K 1A1"
    )
    assert timeline.recent_jobs[0].issue_summary == "Drain garage bloque"
    assert any(
        "Client prefere le texto." in note.body for note in timeline.recent_notes
    )


@pytest.mark.asyncio
async def test_boss_note_and_access_note_updates_round_trip(
    isolated_data_dir, fake_dbmate
):
    provision_tenant_crm(TWILIO_NUMBER)
    crm = get_tenant_crm(TWILIO_NUMBER)
    customer = await crm.upsert_customer_for_phone(
        CUSTOMER_NUMBER, "Jonathan Pelletier"
    )
    location = await crm.upsert_service_location_for_customer(
        customer.customer_id,
        "456 rue Nouvelle, Brossard, J4Z 1A1",
    )

    note_result = await crm.add_internal_note(
        customer.customer_id,
        "Laisser de la place dans l'entree pour le camion.",
        service_location_id=location.location_id,
    )
    access_result = await crm.update_location_access_notes(
        location.location_id,
        "Entrer par la porte laterale.",
    )

    assert note_result.success is True
    assert access_result.success is True

    timeline = await crm.get_customer_timeline(customer.customer_id)
    assert timeline.known_locations[0].access_notes == "Entrer par la porte laterale."
    assert any(
        "Laisser de la place dans l'entree pour le camion." in note.body
        for note in timeline.recent_notes
    )


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


@pytest.mark.asyncio
async def test_create_job_for_booking_is_idempotent_on_calendar_uid(
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

    first_id = await crm.create_job_for_booking(
        customer_id=customer.customer_id,
        service_location_id=location.location_id,
        conversation_id="conv-1",
        calendar_uid="fixture-event",
        issue_summary="Premier resume",
        plumber_notes="Premieres notes.",
        scheduled_start="2030-03-24T09:00:00-04:00",
        scheduled_end="2030-03-24T12:00:00-04:00",
    )
    second_id = await crm.create_job_for_booking(
        customer_id=customer.customer_id,
        service_location_id=location.location_id,
        conversation_id="conv-2",
        calendar_uid="fixture-event",
        issue_summary="Resume mis a jour",
        plumber_notes="Notes mises a jour.",
        scheduled_start="2030-03-25T13:00:00-04:00",
        scheduled_end="2030-03-25T17:00:00-04:00",
    )

    assert second_id == first_id
    history = await crm.get_relevant_customer_history_by_phone(CUSTOMER_NUMBER)
    assert len(history.jobs) == 1
    assert history.jobs[0].issue_summary == "Resume mis a jour"


@pytest.mark.asyncio
async def test_booking_tools_work_without_crm_backend(isolated_data_dir, fake_dbmate):
    calendar = get_tenant_calendar(TWILIO_NUMBER)
    booking_date = (date.today() + timedelta(days=2)).isoformat()
    ctx = cast(
        Any,
        SimpleNamespace(
            deps=AgentDeps(
                calendar=calendar,
                crm=None,
                customer_phone=CUSTOMER_NUMBER,
                conversation_id="conv-no-crm",
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

    updated = await tool_modify_own_booking(
        ctx,
        created.event.uid,
        summary="Drain garage encore bloque",
        service_location="456 rue Nouvelle, Brossard, J4Z 1A1",
        plumber_notes="Nouvelle adresse, meme probleme de drain.",
    )

    assert updated.success is True
    assert updated.event is not None

    cancelled = await tool_cancel_own_booking(ctx, created.event.uid)

    assert cancelled.success is True

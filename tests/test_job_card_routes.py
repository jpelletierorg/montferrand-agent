import asyncio
import sqlite3

from fastapi.testclient import TestClient

from montferrand_agent.calendar import get_tenant_calendar
from montferrand_agent.crm import (
    get_tenant_crm,
    provision_tenant_crm,
    tenant_crm_db_path,
)
from montferrand_agent.server import _maybe_handle_plumber_command, app
from montferrand_agent.tenant import phone_to_filename, save_tenant_profile

from .conftest import CUSTOMER_NUMBER, TEST_PROFILE, TWILIO_NUMBER

ADDRESS = "789 Louis-Hebert, Longueuil, J4K 1A1"
SUMMARY = "Drain de plancher bloque"
NOTES = "Drain de sous-sol bouche apres forte pluie."


def _seed_job(fake_dbmate) -> tuple[int, str, str]:
    del fake_dbmate
    save_tenant_profile(TWILIO_NUMBER, TEST_PROFILE)
    provision_tenant_crm(TWILIO_NUMBER)

    calendar = get_tenant_calendar(TWILIO_NUMBER)
    booking = calendar.create_service_call(
        "2030-03-24",
        "09:00",
        "12:00",
        SUMMARY,
        "Jonathan Pelletier",
        CUSTOMER_NUMBER,
        ADDRESS,
        NOTES,
    )
    assert booking.success is True
    assert booking.event is not None
    event = booking.event

    async def setup() -> tuple[int, str]:
        crm = get_tenant_crm(TWILIO_NUMBER)
        customer = await crm.upsert_customer_for_phone(
            CUSTOMER_NUMBER, "Jonathan Pelletier"
        )
        location = await crm.upsert_service_location_for_customer(
            customer.customer_id,
            ADDRESS,
        )
        job_id = await crm.create_job_for_booking(
            customer_id=customer.customer_id,
            service_location_id=location.location_id,
            conversation_id="conv-123",
            calendar_uid=event.uid,
            issue_summary=SUMMARY,
            plumber_notes=NOTES,
            scheduled_start=event.start_iso,
            scheduled_end=event.end_iso,
        )
        token = await crm.issue_job_token(job_id, ttl_hours=72)
        return job_id, token

    job_id, token = asyncio.run(setup())
    return job_id, token, event.uid


class TestPlumberJobCard:
    def test_job_card_renders(self, isolated_data_dir, fake_dbmate):
        job_id, token, _uid = _seed_job(fake_dbmate)
        del job_id
        tenant_key = phone_to_filename(TWILIO_NUMBER)
        client = TestClient(app)

        response = client.get(f"/p/{tenant_key}/{token}")

        assert response.status_code == 200
        assert "Jonathan Pelletier" in response.text
        assert ADDRESS in response.text
        assert SUMMARY in response.text

    def test_status_update_round_trips_to_crm(self, isolated_data_dir, fake_dbmate):
        job_id, token, _uid = _seed_job(fake_dbmate)
        tenant_key = phone_to_filename(TWILIO_NUMBER)
        client = TestClient(app)

        response = client.post(
            f"/p/{tenant_key}/{token}/status",
            data={"status": "arrived"},
            follow_redirects=True,
        )

        assert response.status_code == 200
        assert "Status updated to Arrived" in response.text

        async def verify() -> None:
            card = await get_tenant_crm(TWILIO_NUMBER).get_job_card(job_id)
            assert card.status == "arrived"
            assert any(
                "Status changed to Arrived." in note.body for note in card.recent_notes
            )

        asyncio.run(verify())

    def test_closeout_marks_follow_up_and_saves_note(
        self, isolated_data_dir, fake_dbmate
    ):
        job_id, token, _uid = _seed_job(fake_dbmate)
        tenant_key = phone_to_filename(TWILIO_NUMBER)
        client = TestClient(app)

        response = client.post(
            f"/p/{tenant_key}/{token}/closeout",
            data={
                "closeout_text": "Need to return with camera.",
                "follow_up_needed": "1",
            },
            follow_redirects=True,
        )

        assert response.status_code == 200
        assert "follow-up flagged" in response.text

        async def verify() -> None:
            card = await get_tenant_crm(TWILIO_NUMBER).get_job_card(job_id)
            assert card.status == "follow_up_needed"
            assert any(
                "Need to return with camera." in note.body for note in card.recent_notes
            )

        asyncio.run(verify())

    def test_invalid_token_returns_not_found(self, isolated_data_dir, fake_dbmate):
        _seed_job(fake_dbmate)
        tenant_key = phone_to_filename(TWILIO_NUMBER)
        client = TestClient(app)

        response = client.get(f"/p/{tenant_key}/bad-token")

        assert response.status_code == 404
        assert "invalid" in response.text.lower()

    def test_edit_details_updates_crm_and_calendar(
        self, isolated_data_dir, fake_dbmate
    ):
        job_id, token, uid = _seed_job(fake_dbmate)
        tenant_key = phone_to_filename(TWILIO_NUMBER)
        client = TestClient(app)

        response = client.post(
            f"/p/{tenant_key}/{token}/edit",
            data={
                "customer_name": "Jonathan P.",
                "service_location": "456 rue Nouvelle, Brossard, J4Z 1A1",
                "issue_summary": "Drain garage encore bloque",
                "plumber_notes": "Verifier le drain et la pente.",
                "access_notes": "Entrer par la porte laterale.",
                "customer_summary": "Client prefere etre appele avant l'arrivee.",
            },
            follow_redirects=True,
        )

        assert response.status_code == 200
        assert "Job details updated" in response.text
        assert "Jonathan P." in response.text
        assert "456 rue Nouvelle, Brossard, J4Z 1A1" in response.text

        async def verify() -> None:
            card = await get_tenant_crm(TWILIO_NUMBER).get_job_card(job_id)
            assert card.customer_name == "Jonathan P."
            assert card.service_location == "456 rue Nouvelle, Brossard, J4Z 1A1"
            assert card.issue_summary == "Drain garage encore bloque"
            assert card.plumber_notes == "Verifier le drain et la pente."
            assert card.access_notes == "Entrer par la porte laterale."
            assert (
                card.customer_summary == "Client prefere etre appele avant l'arrivee."
            )

        asyncio.run(verify())

        events = get_tenant_calendar(TWILIO_NUMBER).list_events(
            "2030-03-24",
            "2030-03-24",
            include_past=True,
        )
        assert len(events.events) == 1
        event = events.events[0]
        assert event.uid == uid
        assert event.customer_name == "Jonathan P."
        assert event.location == "456 rue Nouvelle, Brossard, J4Z 1A1"
        assert event.summary == "Drain garage encore bloque"
        assert event.plumber_notes == "Verifier le drain et la pente."


class TestPlumberNextCommand:
    def test_next_command_returns_summary_and_link(
        self, isolated_data_dir, fake_dbmate, monkeypatch
    ):
        job_id, _token, uid = _seed_job(fake_dbmate)
        del job_id
        del uid
        monkeypatch.setenv("MONTFERRAND_BASE_URL", "https://montferrand.test")
        monkeypatch.setenv("MONTFERRAND_JOB_CARD_TOKEN_TTL_HOURS", "48")

        reply = asyncio.run(
            _maybe_handle_plumber_command(
                TWILIO_NUMBER,
                "next",
                is_boss=True,
            )
        )

        assert reply is not None
        assert "Jonathan Pelletier" in reply
        assert "Open card: https://montferrand.test/p/" in reply

    def test_expired_token_is_rejected(self, isolated_data_dir, fake_dbmate):
        _job_id, token, _uid = _seed_job(fake_dbmate)
        tenant_key = phone_to_filename(TWILIO_NUMBER)
        path = tenant_crm_db_path(TWILIO_NUMBER)
        conn = sqlite3.connect(path)
        try:
            conn.execute(
                "UPDATE job_tokens SET expires_at = '2000-01-01T00:00:00+00:00'"
            )
            conn.commit()
        finally:
            conn.close()

        client = TestClient(app)
        response = client.get(f"/p/{tenant_key}/{token}")

        assert response.status_code == 410
        assert "expired" in response.text.lower()

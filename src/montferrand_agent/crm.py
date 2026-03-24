from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import subprocess
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
from pydantic import BaseModel, Field

from montferrand_agent.config import tenant_db_root_dir
from montferrand_agent.tenant import list_tenants, phone_to_filename


class TenantCrmError(RuntimeError):
    pass


class TenantCrmMissingError(TenantCrmError):
    pass


def crm_migrations_dir() -> Path:
    source_path = Path(__file__).resolve()
    for base in source_path.parents:
        candidate = base / "db" / "crm" / "migrations"
        if candidate.exists():
            return candidate
    return source_path.parents[2] / "db" / "crm" / "migrations"


def tenant_crm_dir(twilio_number: str) -> Path:
    return tenant_db_root_dir() / phone_to_filename(twilio_number)


def tenant_crm_db_path(twilio_number: str) -> Path:
    return tenant_crm_dir(twilio_number) / "crm.sqlite3"


def _dbmate_bin() -> str:
    return os.getenv("MONTFERRAND_DBMATE_BIN", "dbmate").strip() or "dbmate"


def _dbmate_url(db_path: Path) -> str:
    return f"sqlite:{db_path}"


def _migration_version(path: Path) -> str:
    return path.name.split("_", maxsplit=1)[0]


def _read_up_sql(path: Path) -> str:
    lines: list[str] = []
    capture = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("-- migrate:up"):
            capture = True
            continue
        if line.startswith("-- migrate:down"):
            capture = False
            continue
        if capture:
            lines.append(line)
    return "\n".join(lines).strip()


def _run_internal_migrations(db_path: Path, command: str) -> None:
    migrations_dir = crm_migrations_dir()
    migration_files = sorted(migrations_dir.glob("*.sql"))

    if command == "up":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    elif not db_path.exists():
        raise TenantCrmMissingError(f"CRM DB is missing for fallback runner: {db_path}")

    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY)"
    )
    try:
        if command in {"up", "migrate"}:
            applied = {
                row[0] for row in conn.execute("SELECT version FROM schema_migrations")
            }
            for migration_path in migration_files:
                version = _migration_version(migration_path)
                if version in applied:
                    continue
                script = _read_up_sql(migration_path)
                if script:
                    conn.executescript(script)
                conn.execute(
                    "INSERT INTO schema_migrations(version) VALUES (?)",
                    (version,),
                )
            conn.commit()
            return

        if command == "status":
            applied = {
                row[0] for row in conn.execute("SELECT version FROM schema_migrations")
            }
            expected = {_migration_version(path) for path in migration_files}
            if applied != expected:
                missing = ", ".join(sorted(expected - applied)) or "unknown"
                raise TenantCrmError(f"pending migrations: {missing}")
            return

        raise TenantCrmError(f"unsupported internal migration command: {command}")
    finally:
        conn.close()


def _run_dbmate(db_path: Path, command: str) -> None:
    migrations_dir = crm_migrations_dir()
    if not migrations_dir.exists():
        raise TenantCrmError(
            f"CRM migrations directory does not exist: {migrations_dir}"
        )

    cmd = [
        _dbmate_bin(),
        "--migrations-dir",
        str(migrations_dir),
        "--url",
        _dbmate_url(db_path),
    ]
    if command == "status":
        cmd.extend(["status", "--exit-code"])
    else:
        cmd.append(command)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        _run_internal_migrations(db_path, command)
        return
    if result.returncode != 0:
        detail = (
            result.stderr.strip() or result.stdout.strip() or "unknown dbmate error"
        )
        raise TenantCrmError(f"dbmate {command} failed for {db_path}: {detail}")


def provision_tenant_crm(twilio_number: str) -> Path:
    db_path = tenant_crm_db_path(twilio_number)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _run_dbmate(db_path, "up")
    if not db_path.exists():
        raise TenantCrmError(f"CRM DB was not created for {twilio_number}: {db_path}")
    return db_path


def migrate_tenant_crm(twilio_number: str) -> Path:
    db_path = tenant_crm_db_path(twilio_number)
    if not db_path.exists():
        raise TenantCrmMissingError(f"CRM DB is missing for {twilio_number}: {db_path}")
    _run_dbmate(db_path, "migrate")
    return db_path


def verify_tenant_crm(twilio_number: str) -> Path:
    db_path = tenant_crm_db_path(twilio_number)
    if not db_path.exists():
        raise TenantCrmMissingError(f"CRM DB is missing for {twilio_number}: {db_path}")
    _run_dbmate(db_path, "status")
    return db_path


def provision_all_tenant_crm() -> list[Path]:
    return [provision_tenant_crm(phone) for phone, _path in list_tenants()]


def migrate_all_tenant_crm() -> list[Path]:
    return [migrate_tenant_crm(phone) for phone, _path in list_tenants()]


def migrate_existing_tenant_crm() -> tuple[list[Path], list[str]]:
    migrated: list[Path] = []
    missing: list[str] = []
    for phone, _path in list_tenants():
        db_path = tenant_crm_db_path(phone)
        if not db_path.exists():
            missing.append(phone)
            continue
        migrated.append(migrate_tenant_crm(phone))
    return migrated, missing


def ensure_tenant_crm(twilio_number: str) -> Path:
    db_path = tenant_crm_db_path(twilio_number)
    if db_path.exists():
        return migrate_tenant_crm(twilio_number)
    return provision_tenant_crm(twilio_number)


def ensure_existing_tenant_crm() -> tuple[list[Path], list[str]]:
    migrated: list[Path] = []
    provisioned: list[str] = []
    for phone, _path in list_tenants():
        db_path = tenant_crm_db_path(phone)
        if db_path.exists():
            migrated.append(migrate_tenant_crm(phone))
        else:
            provision_tenant_crm(phone)
            provisioned.append(phone)
    return migrated, provisioned


def verify_all_tenant_crm() -> list[Path]:
    return [verify_tenant_crm(phone) for phone, _path in list_tenants()]


def reset_tenant_crm(twilio_number: str) -> Path:
    """Wipe one tenant CRM database and reprovision a fresh empty schema."""

    crm_dir = tenant_crm_dir(twilio_number)
    if crm_dir.exists():
        shutil.rmtree(crm_dir)
    return provision_tenant_crm(twilio_number)


@dataclass(frozen=True)
class CustomerRecord:
    customer_id: int
    display_name: str
    preferred_language: str


class KnownLocation(BaseModel):
    location_id: int
    formatted_address: str
    access_notes: str = ""
    is_primary: bool = False


class CustomerJobSummary(BaseModel):
    job_id: int
    status: str
    issue_summary: str
    service_location: str = ""
    scheduled_start: str | None = None
    scheduled_end: str | None = None
    calendar_uid: str | None = None


class CustomerContextResult(BaseModel):
    success: bool = Field(description="Whether the CRM lookup succeeded.")
    known_customer: bool = Field(
        description="Whether this phone number matched a known customer record."
    )
    customer_id: int | None = Field(default=None)
    customer_name: str = Field(default="")
    preferred_language: str = Field(default="")
    customer_summary: str = Field(default="")
    primary_location: str = Field(default="")
    known_locations: list[KnownLocation] = Field(default_factory=list)
    active_jobs: list[CustomerJobSummary] = Field(default_factory=list)
    recent_jobs: list[CustomerJobSummary] = Field(default_factory=list)
    message: str = Field(default="")


class CustomerHistoryResult(BaseModel):
    success: bool = Field(description="Whether the CRM lookup succeeded.")
    customer_name: str = Field(default="")
    jobs: list[CustomerJobSummary] = Field(default_factory=list)
    message: str = Field(default="")


class CustomerSearchHit(BaseModel):
    customer_id: int
    customer_name: str
    primary_phone: str = ""
    primary_location: str = ""
    active_job_count: int = 0


class CustomerSearchResult(BaseModel):
    success: bool = Field(description="Whether the CRM search succeeded.")
    matches: list[CustomerSearchHit] = Field(default_factory=list)
    message: str = Field(default="")


class JobNote(BaseModel):
    note_id: int
    author_kind: str
    visibility: str
    body: str
    created_at: str


class CustomerTimelineResult(BaseModel):
    success: bool = Field(description="Whether the CRM timeline lookup succeeded.")
    customer_id: int | None = None
    customer_name: str = ""
    phones: list[str] = Field(default_factory=list)
    customer_summary: str = ""
    known_locations: list[KnownLocation] = Field(default_factory=list)
    active_jobs: list[CustomerJobSummary] = Field(default_factory=list)
    recent_jobs: list[CustomerJobSummary] = Field(default_factory=list)
    recent_notes: list[JobNote] = Field(default_factory=list)
    message: str = ""


class JobCardResult(BaseModel):
    success: bool = Field(description="Whether the job lookup succeeded.")
    job_id: int | None = None
    customer_id: int | None = None
    service_location_id: int | None = None
    calendar_uid: str = ""
    status: str = ""
    customer_name: str = ""
    customer_phone: str = ""
    customer_summary: str = ""
    service_location: str = ""
    access_notes: str = ""
    issue_summary: str = ""
    plumber_notes: str = ""
    scheduled_start: str | None = None
    scheduled_end: str | None = None
    recent_jobs: list[CustomerJobSummary] = Field(default_factory=list)
    recent_notes: list[JobNote] = Field(default_factory=list)
    message: str = ""


class CrmMutationResult(BaseModel):
    success: bool = Field(description="Whether the CRM write succeeded.")
    message: str = Field(default="")
    customer_id: int | None = None
    location_id: int | None = None
    note_id: int | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _normalize_address(value: str) -> str:
    return _normalize_text(value.replace("\n", ", "))


def _split_formatted_address(formatted_address: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in formatted_address.split(",") if part.strip()]
    if not parts:
        return "", "", ""
    address_line1 = parts[0]
    city = parts[1] if len(parts) > 1 else ""
    postal_code = ""
    if len(parts) > 2:
        postal_code = parts[2]
    elif len(parts) > 1:
        maybe_postal = re.search(
            r"\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ -]?\d[ABCEGHJ-NPRSTV-Z]\d\b",
            parts[1],
            re.IGNORECASE,
        )
        if maybe_postal:
            postal_code = maybe_postal.group(0).upper()
            city = parts[1].replace(maybe_postal.group(0), "").strip(" ,")
    return address_line1, city, postal_code


def render_customer_context_for_prompt(context: CustomerContextResult | None) -> str:
    if context is None or not context.known_customer:
        return "- No CRM record is available for this sender phone number."

    lines = ["- CRM matched this sender to an existing customer record."]
    if context.customer_name:
        lines.append(f"- Known customer name: {context.customer_name}")
    if context.known_locations:
        if len(context.known_locations) == 1:
            lines.append(
                f"- One saved service address: {context.known_locations[0].formatted_address}"
            )
        else:
            joined = "; ".join(loc.formatted_address for loc in context.known_locations)
            lines.append(f"- Multiple saved service addresses: {joined}")

    if context.active_jobs:
        summaries: list[str] = []
        for job in context.active_jobs[:3]:
            when = (job.scheduled_start or "")[:10] or "date unknown"
            location = f" at {job.service_location}" if job.service_location else ""
            issue = job.issue_summary or "service call"
            summaries.append(f"{when}: {issue}{location} [{job.status}]")
        lines.append(f"- Active jobs/bookings: {'; '.join(summaries)}")
    elif context.recent_jobs:
        summaries = []
        for job in context.recent_jobs[:2]:
            when = (job.scheduled_start or "")[:10] or "date unknown"
            location = f" at {job.service_location}" if job.service_location else ""
            issue = job.issue_summary or "service call"
            summaries.append(f"{when}: {issue}{location} [{job.status}]")
        lines.append(f"- Short past history: {'; '.join(summaries)}")

    lines.append(
        "- Use ONLY these CRM facts or facts returned by CRM tools. If a fact is not present, ask the customer instead of pretending to know it."
    )
    return "\n".join(lines)


@dataclass
class TenantCrmBackend:
    path: Path

    def __post_init__(self) -> None:
        if not self.path.exists():
            raise TenantCrmMissingError(f"CRM DB is missing: {self.path}")

    @asynccontextmanager
    async def connection(self):
        db = await aiosqlite.connect(self.path)
        db.row_factory = aiosqlite.Row
        try:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("PRAGMA busy_timeout = 5000")
            await db.execute("PRAGMA synchronous = NORMAL")
            yield db
        finally:
            await db.close()

    async def healthcheck(self) -> None:
        async with self.connection() as db:
            async with db.execute("SELECT 1") as cursor:
                await cursor.fetchone()

    async def _get_customer_summary_text(self, customer_id: int) -> str:
        sql = "SELECT summary_text FROM customer_summaries WHERE customer_id = ?"
        async with self.connection() as db:
            async with db.execute(sql, (customer_id,)) as cursor:
                row = await cursor.fetchone()
        return "" if row is None else row["summary_text"]

    async def _get_locations_for_customer(
        self, customer_id: int
    ) -> list[KnownLocation]:
        sql = """
        SELECT id, formatted_address, access_notes, is_primary
        FROM service_locations
        WHERE customer_id = ?
        ORDER BY is_primary DESC, updated_at DESC, id DESC
        """
        async with self.connection() as db:
            async with db.execute(sql, (customer_id,)) as cursor:
                rows = await cursor.fetchall()
        return [
            KnownLocation(
                location_id=row["id"],
                formatted_address=row["formatted_address"],
                access_notes=row["access_notes"],
                is_primary=bool(row["is_primary"]),
            )
            for row in rows
        ]

    async def _get_recent_jobs_for_customer(
        self,
        customer_id: int,
        *,
        issue_hint: str | None = None,
        limit: int = 3,
    ) -> list[CustomerJobSummary]:
        sql = """
        SELECT
            j.id,
            j.status,
            j.issue_summary,
            j.scheduled_start,
            j.scheduled_end,
            j.calendar_uid,
            COALESCE(sl.formatted_address, '') AS formatted_address
        FROM jobs AS j
        LEFT JOIN service_locations AS sl ON sl.id = j.service_location_id
        WHERE j.customer_id = ?
        """
        params: list[object] = [customer_id]
        if issue_hint:
            sql += " AND LOWER(j.issue_summary || ' ' || j.plumber_notes) LIKE LOWER(?)"
            params.append(f"%{issue_hint.strip()}%")
        sql += " ORDER BY COALESCE(j.scheduled_start, j.created_at) DESC, j.id DESC LIMIT ?"
        params.append(limit)

        async with self.connection() as db:
            async with db.execute(sql, tuple(params)) as cursor:
                rows = await cursor.fetchall()
        return [
            CustomerJobSummary(
                job_id=row["id"],
                status=row["status"],
                issue_summary=row["issue_summary"],
                service_location=row["formatted_address"],
                scheduled_start=row["scheduled_start"],
                scheduled_end=row["scheduled_end"],
                calendar_uid=row["calendar_uid"],
            )
            for row in rows
        ]

    async def _get_active_jobs_for_customer(
        self,
        customer_id: int,
        *,
        limit: int = 3,
    ) -> list[CustomerJobSummary]:
        sql = """
        SELECT
            j.id,
            j.status,
            j.issue_summary,
            j.scheduled_start,
            j.scheduled_end,
            j.calendar_uid,
            COALESCE(sl.formatted_address, '') AS formatted_address
        FROM jobs AS j
        LEFT JOIN service_locations AS sl ON sl.id = j.service_location_id
        WHERE j.customer_id = ?
          AND j.status NOT IN ('cancelled', 'completed')
        ORDER BY
            CASE j.status
                WHEN 'arrived' THEN 0
                WHEN 'en_route' THEN 1
                WHEN 'booked' THEN 2
                WHEN 'follow_up_needed' THEN 3
                ELSE 4
            END,
            COALESCE(j.scheduled_start, j.created_at) ASC,
            j.id ASC
        LIMIT ?
        """

        async with self.connection() as db:
            async with db.execute(sql, (customer_id, limit)) as cursor:
                rows = await cursor.fetchall()
        return [
            CustomerJobSummary(
                job_id=row["id"],
                status=row["status"],
                issue_summary=row["issue_summary"],
                service_location=row["formatted_address"],
                scheduled_start=row["scheduled_start"],
                scheduled_end=row["scheduled_end"],
                calendar_uid=row["calendar_uid"],
            )
            for row in rows
        ]

    async def _get_recent_notes_for_job(
        self, job_id: int, *, limit: int = 5
    ) -> list[JobNote]:
        sql = """
        SELECT id, author_kind, visibility, body, created_at
        FROM customer_notes
        WHERE job_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """
        async with self.connection() as db:
            async with db.execute(sql, (job_id, limit)) as cursor:
                rows = await cursor.fetchall()
        return [
            JobNote(
                note_id=row["id"],
                author_kind=row["author_kind"],
                visibility=row["visibility"],
                body=row["body"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def _get_recent_notes_for_customer(
        self, customer_id: int, *, limit: int = 5
    ) -> list[JobNote]:
        sql = """
        SELECT id, author_kind, visibility, body, created_at
        FROM customer_notes
        WHERE customer_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """
        async with self.connection() as db:
            async with db.execute(sql, (customer_id, limit)) as cursor:
                rows = await cursor.fetchall()
        return [
            JobNote(
                note_id=row["id"],
                author_kind=row["author_kind"],
                visibility=row["visibility"],
                body=row["body"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def _get_phone_numbers_for_customer(self, customer_id: int) -> list[str]:
        sql = """
        SELECT phone_e164
        FROM customer_phones
        WHERE customer_id = ?
        ORDER BY is_primary DESC, id ASC
        """
        async with self.connection() as db:
            async with db.execute(sql, (customer_id,)) as cursor:
                rows = await cursor.fetchall()
        return [str(row["phone_e164"]) for row in rows if row["phone_e164"]]

    async def _get_job_card_by_id(self, job_id: int) -> JobCardResult:
        sql = """
        SELECT
            j.id,
            j.calendar_uid,
            j.service_location_id,
            j.status,
            j.issue_summary,
            j.plumber_notes,
            j.scheduled_start,
            j.scheduled_end,
            c.id AS customer_id,
            c.display_name,
            COALESCE(
                (
                    SELECT p.phone_e164
                    FROM customer_phones AS p
                    WHERE p.customer_id = c.id
                    ORDER BY p.is_primary DESC, p.id ASC
                    LIMIT 1
                ),
                ''
            ) AS customer_phone,
            COALESCE(sl.formatted_address, '') AS formatted_address,
            COALESCE(sl.access_notes, '') AS access_notes
        FROM jobs AS j
        JOIN crm_customers AS c ON c.id = j.customer_id
        LEFT JOIN service_locations AS sl ON sl.id = j.service_location_id
        WHERE j.id = ?
        LIMIT 1
        """
        async with self.connection() as db:
            async with db.execute(sql, (job_id,)) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return JobCardResult(
                success=False, message=f"No job found for id {job_id}."
            )

        customer_id = int(row["customer_id"])
        customer_summary = await self._get_customer_summary_text(customer_id)
        recent_jobs = [
            job
            for job in await self._get_recent_jobs_for_customer(customer_id, limit=4)
            if job.job_id != job_id
        ][:3]
        recent_notes = await self._get_recent_notes_for_job(job_id)
        return JobCardResult(
            success=True,
            job_id=int(row["id"]),
            customer_id=customer_id,
            service_location_id=(
                None
                if row["service_location_id"] is None
                else int(row["service_location_id"])
            ),
            calendar_uid=row["calendar_uid"] or "",
            status=row["status"] or "",
            customer_name=row["display_name"] or "",
            customer_phone=row["customer_phone"] or "",
            customer_summary=customer_summary,
            service_location=row["formatted_address"] or "",
            access_notes=row["access_notes"] or "",
            issue_summary=row["issue_summary"] or "",
            plumber_notes=row["plumber_notes"] or "",
            scheduled_start=row["scheduled_start"],
            scheduled_end=row["scheduled_end"],
            recent_jobs=recent_jobs,
            recent_notes=recent_notes,
            message="Loaded job card.",
        )

    async def lookup_customer_by_phone(self, phone_e164: str) -> CustomerRecord | None:
        sql = """
        SELECT c.id AS customer_id, c.display_name, c.preferred_language
        FROM customer_phones AS p
        JOIN crm_customers AS c ON c.id = p.customer_id
        WHERE p.phone_e164 = ?
        LIMIT 1
        """
        async with self.connection() as db:
            async with db.execute(sql, (phone_e164,)) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return CustomerRecord(
            customer_id=row["customer_id"],
            display_name=row["display_name"],
            preferred_language=row["preferred_language"],
        )

    async def search_customers(
        self, query: str, *, limit: int = 5
    ) -> CustomerSearchResult:
        clean_query = _normalize_text(query)
        if not clean_query:
            return CustomerSearchResult(
                success=True,
                matches=[],
                message="Search query is empty.",
            )

        like = f"%{clean_query}%"
        sql = """
        SELECT
            c.id,
            c.display_name,
            COALESCE(
                (
                    SELECT p.phone_e164
                    FROM customer_phones AS p
                    WHERE p.customer_id = c.id
                    ORDER BY p.is_primary DESC, p.id ASC
                    LIMIT 1
                ),
                ''
            ) AS primary_phone,
            COALESCE(
                (
                    SELECT sl.formatted_address
                    FROM service_locations AS sl
                    WHERE sl.customer_id = c.id
                    ORDER BY sl.is_primary DESC, sl.updated_at DESC, sl.id DESC
                    LIMIT 1
                ),
                ''
            ) AS primary_location,
            (
                SELECT COUNT(*)
                FROM jobs AS j
                WHERE j.customer_id = c.id
                  AND j.status NOT IN ('cancelled', 'completed')
            ) AS active_job_count
        FROM crm_customers AS c
        WHERE LOWER(c.display_name) LIKE LOWER(?)
           OR EXISTS (
                SELECT 1 FROM customer_phones AS p
                WHERE p.customer_id = c.id AND LOWER(p.phone_e164) LIKE LOWER(?)
           )
           OR EXISTS (
                SELECT 1 FROM service_locations AS sl
                WHERE sl.customer_id = c.id AND LOWER(sl.formatted_address) LIKE LOWER(?)
           )
        ORDER BY active_job_count DESC, c.updated_at DESC, c.id DESC
        LIMIT ?
        """
        async with self.connection() as db:
            async with db.execute(sql, (like, like, like, limit)) as cursor:
                rows = await cursor.fetchall()
        matches = [
            CustomerSearchHit(
                customer_id=int(row["id"]),
                customer_name=row["display_name"] or "",
                primary_phone=row["primary_phone"] or "",
                primary_location=row["primary_location"] or "",
                active_job_count=int(row["active_job_count"] or 0),
            )
            for row in rows
        ]
        return CustomerSearchResult(
            success=True,
            matches=matches,
            message=f"Found {len(matches)} customer(s) matching '{clean_query}'.",
        )

    async def get_customer_timeline(
        self,
        customer_id: int,
        *,
        limit_jobs: int = 5,
        limit_notes: int = 5,
    ) -> CustomerTimelineResult:
        sql = """
        SELECT id, display_name
        FROM crm_customers
        WHERE id = ?
        LIMIT 1
        """
        async with self.connection() as db:
            async with db.execute(sql, (customer_id,)) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return CustomerTimelineResult(
                success=False,
                message=f"No customer found for id {customer_id}.",
            )

        customer_summary = await self._get_customer_summary_text(customer_id)
        phones = await self._get_phone_numbers_for_customer(customer_id)
        locations = await self._get_locations_for_customer(customer_id)
        active_jobs = await self._get_active_jobs_for_customer(
            customer_id, limit=limit_jobs
        )
        recent_jobs = await self._get_recent_jobs_for_customer(
            customer_id, limit=limit_jobs
        )
        recent_notes = await self._get_recent_notes_for_customer(
            customer_id, limit=limit_notes
        )
        return CustomerTimelineResult(
            success=True,
            customer_id=customer_id,
            customer_name=row["display_name"] or "",
            phones=phones,
            customer_summary=customer_summary,
            known_locations=locations,
            active_jobs=active_jobs,
            recent_jobs=recent_jobs,
            recent_notes=recent_notes,
            message=f"Loaded customer timeline for {row['display_name'] or customer_id}.",
        )

    async def get_customer_context_by_phone(
        self, phone_e164: str
    ) -> CustomerContextResult:
        customer = await self.lookup_customer_by_phone(phone_e164)
        if customer is None:
            return CustomerContextResult(
                success=True,
                known_customer=False,
                message="No CRM record exists for this sender phone number.",
            )

        locations = await self._get_locations_for_customer(customer.customer_id)
        active_jobs = await self._get_active_jobs_for_customer(
            customer.customer_id, limit=3
        )
        recent_jobs = []
        if not active_jobs:
            recent_jobs = await self._get_recent_jobs_for_customer(
                customer.customer_id,
                limit=2,
            )
        summary_text = await self._get_customer_summary_text(customer.customer_id)
        primary_location = next(
            (
                location.formatted_address
                for location in locations
                if location.is_primary
            ),
            locations[0].formatted_address if locations else "",
        )
        return CustomerContextResult(
            success=True,
            known_customer=True,
            customer_id=customer.customer_id,
            customer_name=customer.display_name,
            preferred_language=customer.preferred_language,
            customer_summary=summary_text,
            primary_location=primary_location,
            known_locations=locations,
            active_jobs=active_jobs,
            recent_jobs=recent_jobs,
            message=(f"Loaded CRM context for {customer.display_name or phone_e164}."),
        )

    async def get_relevant_customer_history_by_phone(
        self,
        phone_e164: str,
        issue_hint: str | None = None,
        *,
        limit: int = 3,
    ) -> CustomerHistoryResult:
        customer = await self.lookup_customer_by_phone(phone_e164)
        if customer is None:
            return CustomerHistoryResult(
                success=True,
                customer_name="",
                jobs=[],
                message="No CRM history exists for this sender phone number.",
            )

        jobs = await self._get_recent_jobs_for_customer(
            customer.customer_id,
            issue_hint=issue_hint,
            limit=limit,
        )
        if not jobs and issue_hint:
            jobs = await self._get_recent_jobs_for_customer(
                customer.customer_id, limit=limit
            )

        return CustomerHistoryResult(
            success=True,
            customer_name=customer.display_name,
            jobs=jobs,
            message=(
                f"Found {len(jobs)} relevant prior job(s) for {customer.display_name or phone_e164}."
            ),
        )

    async def upsert_customer_for_phone(
        self, phone_e164: str, display_name: str
    ) -> CustomerRecord:
        clean_phone = _normalize_text(phone_e164)
        clean_name = _normalize_text(display_name)
        now = _now_iso()
        existing = await self.lookup_customer_by_phone(clean_phone)
        async with self.connection() as db:
            if existing is None:
                cursor = await db.execute(
                    """
                    INSERT INTO crm_customers(
                        display_name,
                        preferred_language,
                        created_at,
                        updated_at,
                        last_seen_at
                    ) VALUES (?, '', ?, ?, ?)
                    """,
                    (clean_name, now, now, now),
                )
                lastrowid = cursor.lastrowid
                if lastrowid is None:
                    raise TenantCrmError("Failed to create CRM customer record.")
                customer_id = int(lastrowid)
                await db.execute(
                    """
                    INSERT INTO customer_phones(customer_id, phone_e164, is_primary, created_at)
                    VALUES (?, ?, 1, ?)
                    """,
                    (customer_id, clean_phone, now),
                )
                await db.commit()
                return CustomerRecord(
                    customer_id=customer_id,
                    display_name=clean_name,
                    preferred_language="",
                )

            await db.execute(
                """
                UPDATE crm_customers
                SET display_name = ?, updated_at = ?, last_seen_at = ?
                WHERE id = ?
                """,
                (clean_name or existing.display_name, now, now, existing.customer_id),
            )
            await db.commit()
            return CustomerRecord(
                customer_id=existing.customer_id,
                display_name=clean_name or existing.display_name,
                preferred_language=existing.preferred_language,
            )

    async def upsert_service_location_for_customer(
        self, customer_id: int, formatted_address: str
    ) -> KnownLocation:
        clean_address = _normalize_address(formatted_address)
        address_line1, city, postal_code = _split_formatted_address(clean_address)
        now = _now_iso()

        async with self.connection() as db:
            await db.execute(
                "UPDATE service_locations SET is_primary = 0 WHERE customer_id = ?",
                (customer_id,),
            )
            async with db.execute(
                """
                SELECT id, access_notes
                FROM service_locations
                WHERE customer_id = ? AND formatted_address = ?
                LIMIT 1
                """,
                (customer_id, clean_address),
            ) as cursor:
                existing = await cursor.fetchone()

            if existing is None:
                cursor = await db.execute(
                    """
                    INSERT INTO service_locations(
                        customer_id,
                        label,
                        address_line1,
                        city,
                        postal_code,
                        formatted_address,
                        access_notes,
                        is_primary,
                        created_at,
                        updated_at
                    ) VALUES (?, '', ?, ?, ?, ?, '', 1, ?, ?)
                    """,
                    (
                        customer_id,
                        address_line1,
                        city,
                        postal_code,
                        clean_address,
                        now,
                        now,
                    ),
                )
                lastrowid = cursor.lastrowid
                if lastrowid is None:
                    raise TenantCrmError("Failed to create CRM service location.")
                location_id = int(lastrowid)
                access_notes = ""
            else:
                location_id_value = existing["id"]
                if location_id_value is None:
                    raise TenantCrmError("CRM service location row is missing its id.")
                location_id = int(location_id_value)
                access_notes = existing["access_notes"]
                await db.execute(
                    """
                    UPDATE service_locations
                    SET address_line1 = ?, city = ?, postal_code = ?, formatted_address = ?,
                        is_primary = 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        address_line1,
                        city,
                        postal_code,
                        clean_address,
                        now,
                        location_id,
                    ),
                )

            await db.commit()

        return KnownLocation(
            location_id=location_id,
            formatted_address=clean_address,
            access_notes=access_notes,
            is_primary=True,
        )

    async def create_job_for_booking(
        self,
        *,
        customer_id: int,
        service_location_id: int | None,
        conversation_id: str,
        calendar_uid: str,
        issue_summary: str,
        plumber_notes: str,
        scheduled_start: str | None,
        scheduled_end: str | None,
    ) -> int:
        now = _now_iso()
        async with self.connection() as db:
            async with db.execute(
                "SELECT id FROM jobs WHERE calendar_uid = ? LIMIT 1",
                (calendar_uid,),
            ) as cursor:
                existing = await cursor.fetchone()

            if existing is not None:
                await db.execute(
                    """
                    UPDATE jobs
                    SET customer_id = ?,
                        service_location_id = ?,
                        conversation_id = ?,
                        status = 'booked',
                        issue_summary = ?,
                        plumber_notes = ?,
                        scheduled_start = ?,
                        scheduled_end = ?,
                        updated_at = ?,
                        closed_at = NULL
                    WHERE id = ?
                    """,
                    (
                        customer_id,
                        service_location_id,
                        conversation_id,
                        issue_summary,
                        plumber_notes,
                        scheduled_start,
                        scheduled_end,
                        now,
                        int(existing["id"]),
                    ),
                )
                await db.commit()
                return int(existing["id"])

            cursor = await db.execute(
                """
                INSERT INTO jobs(
                    customer_id,
                    service_location_id,
                    conversation_id,
                    calendar_uid,
                    status,
                    issue_summary,
                    plumber_notes,
                    scheduled_start,
                    scheduled_end,
                    created_at,
                    updated_at,
                    closed_at
                ) VALUES (?, ?, ?, ?, 'booked', ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    customer_id,
                    service_location_id,
                    conversation_id,
                    calendar_uid,
                    issue_summary,
                    plumber_notes,
                    scheduled_start,
                    scheduled_end,
                    now,
                    now,
                ),
            )
            await db.commit()
        lastrowid = cursor.lastrowid
        if lastrowid is None:
            raise TenantCrmError("Failed to create CRM job record.")
        return int(lastrowid)

    async def add_job_note(
        self,
        job_id: int,
        body: str,
        *,
        author_kind: str = "plumber",
        visibility: str = "internal",
    ) -> None:
        clean_body = _normalize_text(body)
        if not clean_body:
            return

        async with self.connection() as db:
            async with db.execute(
                "SELECT customer_id, service_location_id FROM jobs WHERE id = ? LIMIT 1",
                (job_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return

            await db.execute(
                """
                INSERT INTO customer_notes(
                    customer_id,
                    service_location_id,
                    job_id,
                    visibility,
                    author_kind,
                    body,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row["customer_id"]),
                    row["service_location_id"],
                    job_id,
                    visibility,
                    author_kind,
                    clean_body,
                    _now_iso(),
                ),
            )
            await db.commit()

    async def add_internal_note(
        self,
        customer_id: int,
        body: str,
        *,
        service_location_id: int | None = None,
        job_id: int | None = None,
    ) -> CrmMutationResult:
        clean_body = _normalize_text(body)
        if not clean_body:
            return CrmMutationResult(success=False, message="Note body is empty.")

        async with self.connection() as db:
            async with db.execute(
                "SELECT id FROM crm_customers WHERE id = ? LIMIT 1",
                (customer_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return CrmMutationResult(
                    success=False,
                    customer_id=customer_id,
                    message=f"No customer found for id {customer_id}.",
                )

            cursor = await db.execute(
                """
                INSERT INTO customer_notes(
                    customer_id,
                    service_location_id,
                    job_id,
                    visibility,
                    author_kind,
                    body,
                    created_at
                ) VALUES (?, ?, ?, 'boss_only', 'boss', ?, ?)
                """,
                (customer_id, service_location_id, job_id, clean_body, _now_iso()),
            )
            await db.commit()
        note_id = cursor.lastrowid
        return CrmMutationResult(
            success=True,
            customer_id=customer_id,
            note_id=None if note_id is None else int(note_id),
            message="Internal note saved.",
        )

    async def upsert_customer_summary(
        self, customer_id: int, summary_text: str
    ) -> CrmMutationResult:
        clean_summary = summary_text.strip()
        now = _now_iso()
        async with self.connection() as db:
            async with db.execute(
                "SELECT id FROM crm_customers WHERE id = ? LIMIT 1",
                (customer_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return CrmMutationResult(
                    success=False,
                    customer_id=customer_id,
                    message=f"No customer found for id {customer_id}.",
                )

            if clean_summary:
                await db.execute(
                    """
                    INSERT INTO customer_summaries(customer_id, summary_text, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(customer_id) DO UPDATE SET
                        summary_text = excluded.summary_text,
                        updated_at = excluded.updated_at
                    """,
                    (customer_id, clean_summary, now),
                )
            else:
                await db.execute(
                    "DELETE FROM customer_summaries WHERE customer_id = ?",
                    (customer_id,),
                )
            await db.commit()
        return CrmMutationResult(
            success=True,
            customer_id=customer_id,
            message="Customer summary updated.",
        )

    async def update_location_access_notes(
        self, location_id: int, access_notes: str
    ) -> CrmMutationResult:
        clean_notes = access_notes.strip()
        async with self.connection() as db:
            cursor = await db.execute(
                """
                UPDATE service_locations
                SET access_notes = ?, updated_at = ?
                WHERE id = ?
                """,
                (clean_notes, _now_iso(), location_id),
            )
            await db.commit()
        if cursor.rowcount == 0:
            return CrmMutationResult(
                success=False,
                location_id=location_id,
                message=f"No service location found for id {location_id}.",
            )
        return CrmMutationResult(
            success=True,
            location_id=location_id,
            message="Access notes updated.",
        )

    async def amend_job_card_fields(
        self,
        job_id: int,
        *,
        customer_name: str,
        service_location: str,
        issue_summary: str,
        plumber_notes: str,
        access_notes: str,
        customer_summary: str,
    ) -> JobCardResult:
        card = await self.get_job_card(job_id)
        if not card.success or card.job_id is None or card.customer_id is None:
            return card

        clean_customer_name = _normalize_text(customer_name) or card.customer_name
        clean_service_location = (
            _normalize_address(service_location) or card.service_location
        )
        clean_issue_summary = _normalize_text(issue_summary) or card.issue_summary
        clean_plumber_notes = plumber_notes.strip() or card.plumber_notes

        customer = await self.upsert_customer_for_phone(
            card.customer_phone,
            clean_customer_name,
        )
        location = await self.upsert_service_location_for_customer(
            customer.customer_id,
            clean_service_location,
        )
        await self.update_location_access_notes(location.location_id, access_notes)
        await self.upsert_customer_summary(customer.customer_id, customer_summary)

        async with self.connection() as db:
            await db.execute(
                """
                UPDATE jobs
                SET service_location_id = ?,
                    issue_summary = ?,
                    plumber_notes = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    location.location_id,
                    clean_issue_summary,
                    clean_plumber_notes,
                    _now_iso(),
                    job_id,
                ),
            )
            await db.commit()

        return await self.get_job_card(job_id)

    async def sync_job_after_booking_modify(
        self,
        *,
        calendar_uid: str,
        customer_name: str,
        service_location: str,
        issue_summary: str,
        plumber_notes: str,
        scheduled_start: str | None,
        scheduled_end: str | None,
    ) -> bool:
        now = _now_iso()
        async with self.connection() as db:
            async with db.execute(
                "SELECT id, customer_id FROM jobs WHERE calendar_uid = ? LIMIT 1",
                (calendar_uid,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return False

        customer_id = int(row["customer_id"])
        job_id = int(row["id"])
        location = await self.upsert_service_location_for_customer(
            customer_id, service_location
        )
        async with self.connection() as db:
            await db.execute(
                """
                UPDATE crm_customers
                SET display_name = ?, updated_at = ?, last_seen_at = ?
                WHERE id = ?
                """,
                (_normalize_text(customer_name), now, now, customer_id),
            )
            await db.execute(
                """
                UPDATE jobs
                SET service_location_id = ?, issue_summary = ?, plumber_notes = ?,
                    scheduled_start = ?, scheduled_end = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    location.location_id,
                    issue_summary,
                    plumber_notes,
                    scheduled_start,
                    scheduled_end,
                    now,
                    job_id,
                ),
            )
            await db.commit()
        return True

    async def mark_job_cancelled_by_calendar_uid(self, calendar_uid: str) -> bool:
        now = _now_iso()
        async with self.connection() as db:
            cursor = await db.execute(
                """
                UPDATE jobs
                SET status = 'cancelled', updated_at = ?, closed_at = ?
                WHERE calendar_uid = ?
                """,
                (now, now, calendar_uid),
            )
            await db.commit()
        return cursor.rowcount > 0

    async def get_next_open_job(self) -> JobCardResult | None:
        now = _now_iso()
        async with self.connection() as db:
            async with db.execute(
                """
                SELECT id
                FROM jobs
                WHERE status NOT IN ('cancelled', 'completed')
                  AND (scheduled_end IS NULL OR scheduled_end >= ?)
                ORDER BY COALESCE(scheduled_start, created_at) ASC, id ASC
                LIMIT 1
                """,
                (now,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return await self._get_job_card_by_id(int(row["id"]))

    async def issue_job_token(self, job_id: int, *, ttl_hours: int = 72) -> str:
        token = secrets.token_urlsafe(24)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        expires_at = (now + timedelta(hours=ttl_hours)).isoformat()
        async with self.connection() as db:
            await db.execute(
                """
                INSERT INTO job_tokens(job_id, token_hash, created_at, expires_at, last_used_at)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (job_id, _hash_token(token), now.isoformat(), expires_at),
            )
            await db.commit()
        return token

    async def get_job_card_by_token(self, token: str) -> JobCardResult | None:
        token_hash = _hash_token(token)
        now = _now_iso()
        async with self.connection() as db:
            async with db.execute(
                """
                SELECT id, job_id, expires_at
                FROM job_tokens
                WHERE token_hash = ?
                LIMIT 1
                """,
                (token_hash,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None
            if row["expires_at"] < now:
                return JobCardResult(
                    success=False, message="This job link has expired."
                )
            await db.execute(
                "UPDATE job_tokens SET last_used_at = ? WHERE id = ?",
                (now, int(row["id"])),
            )
            await db.commit()
        return await self._get_job_card_by_id(int(row["job_id"]))

    async def update_job_status(self, job_id: int, status: str) -> JobCardResult:
        now = _now_iso()
        closed_at = now if status in {"completed", "cancelled"} else None
        async with self.connection() as db:
            cursor = await db.execute(
                """
                UPDATE jobs
                SET status = ?, updated_at = ?, closed_at = COALESCE(?, closed_at)
                WHERE id = ?
                """,
                (status, now, closed_at, job_id),
            )
            await db.commit()
        if cursor.rowcount == 0:
            return JobCardResult(
                success=False, message=f"No job found for id {job_id}."
            )
        return await self._get_job_card_by_id(job_id)

    async def close_job(
        self,
        job_id: int,
        closeout_text: str,
        *,
        follow_up_needed: bool = False,
    ) -> JobCardResult:
        await self.add_job_note(job_id, closeout_text, author_kind="plumber")
        return await self.update_job_status(
            job_id,
            "follow_up_needed" if follow_up_needed else "completed",
        )

    async def get_job_card(self, job_id: int) -> JobCardResult:
        return await self._get_job_card_by_id(job_id)


def get_tenant_crm(twilio_number: str) -> TenantCrmBackend:
    return TenantCrmBackend(path=tenant_crm_db_path(twilio_number))


def maybe_get_tenant_crm(twilio_number: str) -> TenantCrmBackend | None:
    try:
        return get_tenant_crm(twilio_number)
    except TenantCrmMissingError:
        return None


def get_tenant_crm_by_key(tenant_key: str) -> TenantCrmBackend:
    return TenantCrmBackend(path=tenant_db_root_dir() / tenant_key / "crm.sqlite3")

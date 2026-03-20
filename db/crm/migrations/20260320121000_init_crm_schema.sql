-- migrate:up
CREATE TABLE crm_customers (
  id INTEGER PRIMARY KEY,
  display_name TEXT NOT NULL DEFAULT '',
  preferred_language TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL
);

CREATE TABLE customer_phones (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  phone_e164 TEXT NOT NULL,
  is_primary INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  FOREIGN KEY (customer_id) REFERENCES crm_customers(id) ON DELETE CASCADE,
  UNIQUE (phone_e164)
);

CREATE TABLE service_locations (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  label TEXT NOT NULL DEFAULT '',
  address_line1 TEXT NOT NULL,
  city TEXT NOT NULL,
  postal_code TEXT NOT NULL DEFAULT '',
  formatted_address TEXT NOT NULL,
  access_notes TEXT NOT NULL DEFAULT '',
  is_primary INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (customer_id) REFERENCES crm_customers(id) ON DELETE CASCADE
);

CREATE TABLE jobs (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  service_location_id INTEGER,
  conversation_id TEXT NOT NULL DEFAULT '',
  calendar_uid TEXT,
  status TEXT NOT NULL,
  issue_summary TEXT NOT NULL DEFAULT '',
  plumber_notes TEXT NOT NULL DEFAULT '',
  scheduled_start TEXT,
  scheduled_end TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  closed_at TEXT,
  FOREIGN KEY (customer_id) REFERENCES crm_customers(id) ON DELETE CASCADE,
  FOREIGN KEY (service_location_id) REFERENCES service_locations(id) ON DELETE SET NULL
);

CREATE TABLE customer_notes (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  service_location_id INTEGER,
  job_id INTEGER,
  visibility TEXT NOT NULL,
  author_kind TEXT NOT NULL,
  body TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (customer_id) REFERENCES crm_customers(id) ON DELETE CASCADE,
  FOREIGN KEY (service_location_id) REFERENCES service_locations(id) ON DELETE SET NULL,
  FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE SET NULL
);

CREATE TABLE customer_summaries (
  customer_id INTEGER PRIMARY KEY,
  summary_text TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (customer_id) REFERENCES crm_customers(id) ON DELETE CASCADE
);

CREATE INDEX idx_customer_phones_phone ON customer_phones (phone_e164);
CREATE INDEX idx_service_locations_customer_updated ON service_locations (customer_id, updated_at DESC);
CREATE INDEX idx_jobs_customer_created ON jobs (customer_id, created_at DESC);
CREATE UNIQUE INDEX idx_jobs_calendar_uid ON jobs (calendar_uid) WHERE calendar_uid IS NOT NULL;
CREATE INDEX idx_customer_notes_customer_created ON customer_notes (customer_id, created_at DESC);

-- migrate:down
DROP INDEX IF EXISTS idx_customer_notes_customer_created;
DROP INDEX IF EXISTS idx_jobs_calendar_uid;
DROP INDEX IF EXISTS idx_jobs_customer_created;
DROP INDEX IF EXISTS idx_service_locations_customer_updated;
DROP INDEX IF EXISTS idx_customer_phones_phone;
DROP TABLE IF EXISTS customer_summaries;
DROP TABLE IF EXISTS customer_notes;
DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS service_locations;
DROP TABLE IF EXISTS customer_phones;
DROP TABLE IF EXISTS crm_customers;

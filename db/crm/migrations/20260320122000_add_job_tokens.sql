-- migrate:up
CREATE TABLE job_tokens (
  id INTEGER PRIMARY KEY,
  job_id INTEGER NOT NULL,
  token_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  last_used_at TEXT,
  FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
  UNIQUE (token_hash)
);

CREATE INDEX idx_job_tokens_job ON job_tokens (job_id);
CREATE INDEX idx_job_tokens_expires ON job_tokens (expires_at);

-- migrate:down
DROP INDEX IF EXISTS idx_job_tokens_expires;
DROP INDEX IF EXISTS idx_job_tokens_job;
DROP TABLE IF EXISTS job_tokens;

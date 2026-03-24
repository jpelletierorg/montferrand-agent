# AGENTS.md

Guidance for coding agents working in `/Users/jopela/Projects/montferrand-agent`.

## Scope

- This repo has two main surfaces:
  - Python backend and CLI in `src/montferrand_agent/`
  - Astro marketing site in `web/`
- Prefer small, targeted changes that follow the existing structure.
- Preserve user changes already in the worktree; do not revert unrelated edits.

## Rule Files

- No prior root `AGENTS.md` existed when this file was written.
- No `.cursorrules` file exists.
- No `.cursor/rules/` directory exists.
- No `.github/copilot-instructions.md` file exists.
- No `.github/` directory exists.
- Follow this file and the conventions already present in source.

## Repo Map

- `src/montferrand_agent/` - app code, CLI, FastAPI server, agent logic, CRM, calendar, ops.
- `tests/` - pytest suite for backend and CLI.
- `db/crm/migrations/` - SQLite/dbmate CRM migrations.
- `web/` - Astro site with Tailwind v4.
- `Taskfile.yml` - task wrappers for web build/deploy.
- `DESIGN.md` - brand and frontend direction for the marketing site.

## Tooling And Runtime

- Python package manager/build tool: `uv`.
- Python requirement: `>=3.13`.
- Node package manager: `npm`.
- Web Node requirement: `>=22.12.0`.
- Prefer `uv run ...` instead of activating the venv manually.
- Prefer `npm` in `web/`; there is a checked-in `web/package-lock.json`.
- Persistent runtime data is driven by `MONTFERRAND_DATA_DIR`.
- Twilio, model provider, admin token, and timezone config are env-driven.
- Do not hardcode secrets, API keys, tenant numbers, or absolute deployment paths.

## Build, Lint, And Test Commands

### Python / backend (run from repo root)

```sh
uv sync --dev
uv build
uv run pytest -q
uv run pytest tests/test_server.py::TestHealth::test_health_check -q
uv run pytest tests/test_server.py -k health_check -q
uv run montferrand serve
uv run montferrand cli
uv run montferrand evals
uv run locust -f tests/load/locustfile.py --host http://127.0.0.1:8080
```

Notes:

- The most reliable single-test form is `uv run pytest path/to/test_file.py::TestClass::test_name -q`.
- Use `-k <expr>` when you only know part of the test name.
- There is no repo-defined Python lint or typecheck command today.
- There is no Ruff, Black, isort, Flake8, mypy, or pyright config in the repo.

### Web / Astro (run from `web/` unless noted)

```sh
npm install
npm run dev
npm run build
npm run preview
npm run astro -- check
```

From repo root:

```sh
task build:web
task deploy:web
```

Notes:

- `task build:web` just runs `npm run build` inside `web/`.
- `task deploy:web` builds first, then runs `node scripts/deploy-web.mjs`.
- There is no `lint` script and no web test runner configured.
- `npm run astro -- check` currently prompts to install `@astrojs/check` and `typescript`.

## Python Code Style

- Keep the module docstring first, then `from __future__ import annotations`.
- Group imports as stdlib, third-party, then local package imports.
- For long local import lists, use parenthesized multiline imports with trailing commas.
- Follow the existing Black-like formatting style even though Black is not configured.
- Use 4-space indentation and double quotes.
- Prefer `pathlib.Path` over raw path strings.
- Prefer built-in generics like `list[str]`, `dict[str, int]`, and unions like `str | None`.
- Add return type annotations to public functions and most helpers.
- Use `Literal`, `TypeAlias`, dataclasses, and Pydantic models when they match existing patterns.
- This package is typed (`src/montferrand_agent/py.typed` exists); avoid introducing untyped public APIs unnecessarily.
- Use `snake_case` for functions, variables, and modules.
- Use `PascalCase` for classes and Pydantic models.
- Docstrings are common and should explain non-obvious behavior, invariants, and storage layout.

## Python Error Handling And Boundaries

- Fail fast on missing required env vars; `RuntimeError` is the common choice for misconfiguration.
- Prefer domain-specific exceptions where the module already has them, e.g. `ConversationError`, `TenantCrmError`, `TenantNotFoundError`.
- At FastAPI boundaries, convert expected failures to `HTTPException` with clear status codes.
- At CLI boundaries, convert fatal errors into clean exits instead of raw stack traces.
- Preserve exception chaining with `raise ... from exc` when wrapping errors.
- Log unexpected exceptions with `logger.exception(...)`.
- Do not silently swallow data corruption, missing config, or failed external calls.

## Project-Specific Backend Conventions

- Keep env-driven configuration centralized; `config.py` is the source of truth for persistent directories.
- Tenant storage is keyed by Twilio phone number and hashed filenames; preserve that scheme.
- Conversation history is persisted as tenant-scoped NDJSON.
- CRM state is SQLite-backed and migrations live in `db/crm/migrations/`.
- When changing CRM schema or tenant storage, update code, migrations, and tests together.
- The server intentionally validates Twilio signatures and fails startup if auth config is missing.
- Async/background behavior matters in `server.py`; preserve graceful shutdown and in-flight task tracking.
- If you change prompt behavior in `agent.py`, inspect `evals.py`, `tool_use_fixtures.py`, and related tests in the same pass.

## Testing Conventions

- Tests are written with `pytest` and grouped in `tests/test_*.py`.
- Class-based grouping such as `class TestHealth:` is common and acceptable.
- Reuse fixtures from `tests/conftest.py` before inventing new setup helpers.
- `monkeypatch`, `tmp_path`, `AsyncMock`, and `unittest.mock.patch` are used heavily.
- Isolate filesystem state by setting `MONTFERRAND_DATA_DIR` to a temp directory.
- Use `fake_dbmate` when tests need CRM migrations without invoking external tooling.
- Prefer deterministic tests with patched network/model/Twilio boundaries.
- When editing server webhook behavior, test both happy paths and HTTP/auth/signature failures.

## Web / Astro Style

- Use relative imports in frontmatter.
- Component filenames use `PascalCase.astro`; page routes stay route-oriented like `index.astro` and `confidentialite.astro`.
- Frontmatter/inline JS follows the existing style: 2-space indentation, single quotes, semicolons.
- Define typed `Props` interfaces in frontmatter when components accept props.
- The site is Tailwind-first; reuse design tokens in `web/src/styles/global.css`.
- Prefer shared tokens and utility classes over one-off inline styles.
- Keep the current French-first, plainspoken marketing voice.
- Preserve the visual direction in `DESIGN.md`; avoid generic SaaS styling.
- Do not add a client framework for simple interactions that already fit plain Astro plus small scripts.

## Practical Agent Checklist

- Read the relevant module and its tests before editing.
- Check whether the change affects backend, web, or both.
- Run the narrowest useful test first, then the broader suite if the change warrants it.
- If you touch `web/`, run `npm run build`; if you touch Python behavior, run at least the affected pytest target.
- If you add new commands, tooling, or rule files, update this `AGENTS.md`.

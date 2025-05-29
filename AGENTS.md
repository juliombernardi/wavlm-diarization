# AGENTS.md – Project Ground‑Rules for WavLM‑MSDD

> **Purpose**
> A one‑page contract between humans and AI agents (e.g. OpenAI Codex) that explains *how* we write, test, lint and ship code in this repository. Keep it updated as the project evolves.

---

## 1 Environment baseline

| Item                 | Value                                                                 |
| -------------------- | --------------------------------------------------------------------- |
| Primary language     | **Python 3.13** (CPython 3.13.2)                                      |
| Target OS            | Ubuntu 20.04 LTS (focal)                                              |
| Min supported Python | 3.10 (for downstream users on older distros)                          |
| Package manager      | `pip` + `requirements.txt` (runtime) / `pyproject.toml` (dev tooling) |
| Virtual‑env          | `.venv/` in repo root (git‑ignored)                                   |

---

## 2 Repository layout (TL;DR)

```
├── src/                 ← importable code (PEP 420 namespace package)
│   └── wavlmmsdd/       ← core library
├── tests/               ← pytest suites (create as project grows)
├── notebook/            ← exploratory notebooks (non‑prod)
├── .data/               ← example audio assets (small, public domain)
├── .temp/               ← throw‑away intermediates (git‑ignored)
└── tools/, scripts/     ← one‑off utilities (planned)
```

*Anything generated at runtime must write into `.temp/`.*

---

## 3 Coding conventions

* **Style & linting:**  Ruff (`ruff check .`) with formatter (`ruff format .`).
  *Config lives in `pyproject.toml`; fail on severity ≥ `E`.*
* **Typing:**  Use `from __future__ import annotations` + standard `typing` types.
  *Mypy may be added later; keep stubs ready.*
* **Imports:**  Absolute (`wavlmmsdd.audio.diarization`) inside `src/`; relative only inside same sub‑package.
* **Docstrings:**  Google style.
* **Public API:**  Re‑export via `wavlmmsdd/__init__.py`; everything else is internal.
* **No secrets in code.**  Credentials must come from env‑vars or a mounted secrets file.

---

## 4 Testing & coverage

* Framework: **pytest ≥ 8** with fixtures in `tests/conftest.py`.
* Minimum coverage gate: **90 % lines** on `src/` (`pytest --cov=src`).
* Slow / integration tests mark with `@pytest.mark.slow`; excluded from default run.
* Data‑dependent tests must use the sample files in `.data/` or synthesised dummy data.

---

## 5 Continuous integration

GitHub Actions workflow (`.github/workflows/ci.yml`) must:

1. Set up Python 3.13.
2. `pip install -U ruff pytest pytest-cov`.
3. `ruff check .` – fails on any error.
4. `pytest -q --cov=src` – fails below 90 %.

Branch protection requires this job to pass before merge into `main`.

---

## 6 Branching & commits

* **Main branch:** `main` (release‑ready).
* **Feature branches:** `feat/<slug>` or `fix/<slug>`.
* **Commit messages:** Conventional Commits (`feat:`, `fix:`, `refactor:`, `ci:`, etc.).
* **Pull requests:** Draft until CI is green + at least one review.

---

## 7 AI agent usage guidelines

* **Prompt tips** (Codex CLI/Web):

  * *Generate tests:*  `"Write pytest tests for src/wavlmmsdd/audio/*.py (≥90 % coverage)"`.
  * *Refactor & lint:* `--auto-edit "Vectorise loops in wavlmmsdd/audio/utils/utils.py; fix Ruff issues"`.
* Always let Codex run in *Auto‑edit* mode inside a sandbox; it will execute Ruff + pytest.
* Review diffs before approval—AI suggestions must obey all sections of this document.

---

## 8 Ignored artefacts

`.gitignore` and `.codexignore` **must** include:

```
.venv/
.temp/
**/__pycache__/
*.py[cod]
*.wav  # large raw audio not suitable for repo
```

---

## 9 Licensing & attribution

* Project licence: see `LICENSE` (Apache‑2.0).
  New source files must include the standard Apache header (see `src/wavlmmsdd/__init__.py`).
* Third‑party code snippets require attribution comment with URL and licence.

---

## 10 Road‑map snapshots (edit as you go)

* [ ] Migrate remaining scripts in repo root into `src/` package.
* [ ] Add `tests/` coverage for `audio/diarization/diarize.py`.
* [ ] Publish wheels to PyPI.

---

*Last updated:* <!-- 2025‑05‑29 -->

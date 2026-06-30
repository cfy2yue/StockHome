# Git And Collaboration

Updated: 2026-07-01

## Project Identity

- Server directory: `/data/cyx/1030/stock`
- GitHub target: `https://github.com/cfy2yue/StockHome`
- SSH entry: `ssh cyx-server-proxy-cfy`, then `cd /data/cyx/1030/stock`
- Windows clone path may be `E:\stock` or another local clone chosen by CC
- Shared biological dataset root is not used by default

Current audit note: `/data/cyx/1030/stock` is now a local Git worktree on branch
`main` with origin `https://github.com/cfy2yue/StockHome.git`. Project-level
`.gitignore` exists and excludes raw data, caches, reports, runs, secrets,
logs, `.conda/`, and token files. Remote `main` was initialized from this
server workspace on 2026-07-01.

## Codex Alone

Codex works on the server project directory when server caches, credentials,
large reports, backtests, or long-running agents are involved. It may edit code,
docs, configs, and tests when the user asks for implementation.

Do not print, copy, or commit API keys or tokens. Any substantial backtest or
DeepSeek/paid-model run must have a clear goal, leakage boundary, and output
path. Preserve `reports/`, `data/`, `memory/`, `book_skills/`, and `runs/`
unless the user approves a manifest-based archive or cleanup.

## CC Alone

CC should clone `https://github.com/cfy2yue/StockHome` on Windows for direction
audit, documentation cleanup, user-manual edits, goal/plan refinement, and code
review. Small local tests are fine when they do not require server-only caches,
credentials, or large backtests.

If server inspection is required:

```bash
ssh cyx-server-proxy-cfy
cd /data/cyx/1030/stock
```

CC should not edit server code or run server tasks unless Codex is paused for
the scoped file, branch, or task.

## Codex Plus CC

CC is best for product-direction checks, user-facing wording, evidence audit,
goal tightening, and independent critique. Codex is best for server execution,
large local caches, implementation, tests, and result integration.

Record ownership in `docs/HANDOFF.md`, `goal.md`, or a dated doc section before
parallel work. Avoid editing the same code file at the same time. Markdown can
be parallelized when dated append sections are used.

For the engineered local/remote workflow, read
`docs/CC_CODEX_COOPERATION_PROTOCOL.md` before starting parallel work.

## Git Hygiene

Track candidates:

- `README.md`, `AGENTS.md`, `goal.md`, `docs/*.md`
- `src/`, `scripts/`, `tests/`, `config/`, `examples/`
- small YAML/JSON config and user-facing manuals

Do not track by default:

- `data/`, `reports/`, `runs/`, `logs/`, `memory/ledger` dumps, `.conda/`,
  raw book PDFs, caches, Excel outputs, secrets, tokens, API keys,
  `docs/local_archive/`
- `ds_api.txt`, `tushare_token.txt`, `.env`, `.env.*`, `.key`, `.pem`

Before any commit, run a secret/future-field hygiene check appropriate to the
changed files and inspect `git status --short` for accidental large artifacts.

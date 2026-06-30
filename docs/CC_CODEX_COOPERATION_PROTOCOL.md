# CC/Codex Cooperation Protocol

Updated: 2026-07-01

This file is the durable cooperation protocol for the StockHome repository. It
should be read by CC before local work and included in remote Codex handoff
prompts when server execution is requested.

## Ownership

- CC/Cursor owns local coordination: product-direction audit, financial-safety
  wording, leakage review, goal refinement, docs, prompts, code review, small
  safe local patches, GitHub sync, and user-facing planning.
- Remote Codex owns server execution: credential-backed data/API runs, large
  backtests, implementation on the server, result integration, and progress
  reports from exact server paths.
- Do not edit the same code file from CC and remote Codex at the same time.
  Record ownership in `goal.md`, `docs/HANDOFF.md`, or a dated handoff.

## Required Startup Check

Before non-trivial work:

```powershell
git -C E:\cc_workspace\stock fetch origin --prune
git -C E:\cc_workspace\stock status -sb
git -C E:\cc_workspace\stock rev-list --left-right --count HEAD...origin/main
ssh cyx-server-proxy-cfy "cd /data/cyx/1030/stock && git fetch origin --prune && git status -sb && git rev-list --left-right --count HEAD...origin/main"
```

If local, GitHub, and server differ, sync first or report the divergence. Do not
start broad edits from an old base.

## Local Scope

Allowed locally:

- read/review docs and source;
- audit financial-safety language, future-field leakage, stale assumptions, and
  user-facing workflow clarity;
- refine goals, prompts, and Markdown plans;
- prepare small doc/code patches when the user asks.

Not local by default:

- token-consuming API/model calls, live watch loops, broad backtests, or large
  data/report rebuilds;
- modifying `reports/`, `data/`, `runs/`, `logs/`, `memory/`, book assets,
  caches, or credentials.

## Remote Codex Scope

When server/API/backtest work is needed, CC should hand remote Codex a concrete
prompt with:

- objective and success criteria;
- files already inspected;
- input/leakage boundary;
- credential/cost boundary;
- files/tasks Codex owns;
- files/tasks CC owns;
- permissions and stop rules;
- expected output paths.

Use `gpt-5.5` for hard implementation/research planning and `gpt-5.4-mini` for
cheap status/doc checks. Current remote CLI order is `codex -a never exec ...`.

## Git Rule

Document locally, commit locally, push to GitHub only when requested or needed
for server sync. After push, update the server with:

```bash
cd /data/cyx/1030/stock
git pull --ff-only
```

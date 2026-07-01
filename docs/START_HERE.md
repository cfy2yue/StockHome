# Start Here

Updated: 2026-07-01

This project is the A-share research assistant. It should produce clear,
actionable, evidence-grounded research support such as buy, sell, add, reduce,
hold, wait, or collect-more-data recommendations. It must not place orders,
connect to broker APIs, or promise returns.

## Entry

- Server directory: `/data/cyx/1030/stock`
- GitHub target: `https://github.com/cfy2yue/StockHome`
- SSH entry for the user/remote environment: `ssh cyx-server-cfy`, then
  `cd /data/cyx/1030/stock`

## Read First

Current manual local-audit workflow:

1. `local_goal.md`
2. `local_audit.md`
3. `local_suggestion.md`
4. `goal.md`
5. `AGENTS.md`
6. `docs/GIT_AND_COLLABORATION.md`
7. `docs/GITHUB_FILE_MAP.md`
8. `docs/USER_GUIDE.md`
9. `docs/PROJECT_ENTRY.md`
10. `docs/DIRECTORY_MAP.md`
11. `docs/HANDOFF.md`

The three `local_*.md` files are the current local-audit packet. If
`local_goal.md` does not contain a filled `Exact Next Task`, remote Codex is not
in active goal mode yet.

Files under `docs/archive/legacy_auto_coordination_20260701/` are historical
evidence only. Do not treat them as active instructions.

## Manual Local/Remote Boundary

Local CC/Codex reviews the GitHub clone, audits goals, checks data/metric/code
risks, optionally runs small non-destructive checks, and updates
`local_goal.md`, `local_audit.md`, and `local_suggestion.md`.

Remote Codex executes only after the user manually pulls GitHub and starts a
goal. Remote Codex should read the three `local_*.md` files plus `goal.md`,
record decisions and results, and output a structured local-audit request when
blocked.

## Remote Trigger Protocol

When the user types `本地审计指令`, remote Codex must pause new large work,
avoid remote git commit/push/reset/delete operations in that status-export
turn, and output a structured
`LOCAL_AUDIT_REQUEST` containing project path, branch, HEAD, dirty state, files
read, final target, current route, recent commands, changed files, metrics,
best/negative/anomalous results, suspected bottlenecks, at least three
directions for local audit, and suggested updates to `local_goal.md`,
`local_audit.md`, and `local_suggestion.md`.

This does not forbid local CC/Codex from later committing and pushing updated
`local_goal.md`, `local_audit.md`, and `local_suggestion.md`; that push is the
normal way remote Codex receives the next local-audited packet.

When the user types `本地审计结束`, remote Codex must run `git fetch origin` and
`git pull --ff-only`, read `README.md`, `docs/START_HERE.md`, `goal.md`,
`local_goal.md`, `local_audit.md`, and `local_suggestion.md`, then summarize the
next task, resource limits, stop rules, and any document conflicts. If
`local_goal.md` still has no filled `Exact Next Task`, remote Codex must wait
instead of starting goal work.

Manual goal prompt:

```text
目标与路线：local_goal.md
本地审计：local_audit.md
本地建议：local_suggestion.md
资源限制：<fill in this round's limits>
```

## Product Boundary

- Single-stock watchlist/research is the P0 user-facing path.
- Multi-stock comparison is a P1 support path.
- Portfolio/backtest research is P2 and must not override P0/P1 user-facing
  conclusions.
- Reports must state action, position/risk limits, evidence, counter-evidence,
  triggers, and review conditions.
- Do not leak future labels, GT fields, or after-the-fact outcomes into
  evidence packs.

## Do Not Touch Without Approval

- API keys, tokens, broker credentials, and local environment files.
- Raw books, BookSkill sources, critical caches, and backtest evidence.
- `reports/`, `runs/`, `memory/`, and `docs/local_archive/` destructive edits.
- Any live trading or broker integration.

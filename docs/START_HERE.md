# Start Here

Updated: 2026-07-02

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
4. `remote_decision.md`
5. `goal.md`
6. `AGENTS.md`
7. `docs/GIT_AND_COLLABORATION.md`
8. `docs/GITHUB_FILE_MAP.md`
9. `docs/USER_GUIDE.md`
10. `docs/PROJECT_ENTRY.md`
11. `docs/DIRECTORY_MAP.md`
12. `docs/HANDOFF.md`

The three `local_*.md` files are the local-authored remote execution packet.
Remote Codex reads them to execute a user-started goal, but must not edit them.
Local CC/Codex updates these files between remote runs and pushes them to
GitHub. If `local_goal.md` does not contain a filled `Exact Next Task`, remote
Codex waits rather than inventing work.

`remote_decision.md` is the remote-side Chinese decision log. Remote Codex may
create or append it during goal execution. Use it to record AUTONOMOUS_DECISION,
ROUTE_PIVOT, weak/negative metrics, subagent advice, and why remote continued
instead of stopping.

Files under `docs/archive/legacy_auto_coordination_20260701/` are historical
evidence only. Do not treat them as active instructions.

## Manual Local/Remote Boundary

Local CC/Codex reviews the GitHub clone, audits goals, checks data/metric/code
risks, optionally runs small non-destructive checks, and authors/updates
`local_goal.md`, `local_audit.md`, and `local_suggestion.md`.

Remote Codex executes only after the user manually pulls GitHub and starts a
goal. Remote Codex should read the three `local_*.md` files, `remote_decision.md`,
and `goal.md`; execute from the filled `Exact Next Task`; record decisions and
results in RUN_STATUS/reports and `remote_decision.md`; and continue toward the
final goal unless it reaches ACHIEVED, a hard BLOCKED boundary, or user
interruption.

`LOCAL_AUDIT_REQUEST` is a soft audit marker, not a long-goal stop reason. If a
target60, frozen-ranker, signal-family, or small-model route misses its gate,
remote Codex should record evidence in `remote_decision.md`, design a new safe
route or bounded diagnostic inside the resource/safety limits, optionally use
subagents, and continue. Mark hard BLOCKED only if all reasonable next routes
require changing the final target, resource boundary, data/credential
permission, broker/live-trading boundary, or destructive operation.

## Remote Trigger Protocol

When the user types `本地审计指令`, remote Codex must pause new large work,
avoid remote git commit/push/reset/delete operations in that status-export
turn, and output a structured `LOCAL_AUDIT_REQUEST` containing project path,
branch, HEAD, dirty state, files read, final target, current route, recent
commands, changed files, metrics, best/negative/anomalous results, suspected
bottlenecks, at least three directions for local audit, and suggested updates
to `local_goal.md`, `local_audit.md`, and `local_suggestion.md`.

This trigger is user-requested manual audit. It does not mean ordinary negative
results during goal mode should stop the goal.

When the user types `本地审计结束`, remote Codex must run `git fetch origin` and
`git pull --ff-only`, read `README.md`, `docs/START_HERE.md`, `goal.md`,
`local_goal.md`, `local_audit.md`, `local_suggestion.md`, and
`remote_decision.md`, then summarize the next task, resource limits, hard stop
rules, output paths, remote decision history, and any document conflicts. It
must not edit the three `local_*.md` files. If `local_goal.md` still has no
filled `Exact Next Task`, remote Codex must wait instead of starting goal work.

Manual goal prompt:

```text
目标与路线：local_goal.md
本地审计：local_audit.md
本地建议：local_suggestion.md
远端决策日志：remote_decision.md
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

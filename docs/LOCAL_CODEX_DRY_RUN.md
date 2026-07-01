# Local Codex Dry Run

Updated: 2026-07-01

This file records the intended local-Codex dry-run workflow for the StockHome
repo cloned from:

```text
https://github.com/cfy2yue/StockHome.git
```

## Purpose

This is a workflow and documentation smoke test, not the main future local
workflow. The expected long-term local collaborator is CC/Cursor. Local Codex may
be used briefly to validate startup docs, Git hygiene, SSH handoff, and
server/API-process boundaries.

## Local Codex May Do

- Read and summarize docs/source.
- Audit financial-safety language, future-field leakage risks, and stale
  instructions.
- Draft goals, plans, review reports, and small documentation patches.
- Prepare handoff notes for server Codex.
- Run tiny local static checks if dependencies are already present.

## Local Codex Must Not Do

- Run token-consuming DeepSeek/API jobs, broad backtests, or large data
  processing without explicit user approval.
- Pretend local Windows has server caches, reports, ledgers, or credentials.
- Commit or print tokens, API keys, secrets, runs, reports, logs, venvs, or
  local archives.
- Edit the same code files as server Codex without an explicit ownership note.

## Remote Cooperation Dry Run

Use SSH only for lightweight status checks unless the user explicitly asks for
server execution:

```powershell
ssh cyx-server-cfy "cd /data/cyx/1030/stock && git status -sb && git remote -v"
```

When handing a server/API/backtest task to Codex, include:

- objective;
- files inspected locally;
- exact requested server action;
- input and leakage boundary;
- credential/cost boundary;
- expected outputs;
- acceptance check.

## Remote Process Monitoring

Do not start a remote process for this dry run. For future real server jobs:

- use detached `tmux`, `nohup`, or a scheduler for long work;
- document command, start time, PID/session, log path, expected outputs, and
  stop rule;
- check logs sparingly;
- report exact server paths, not copied log dumps;
- never print API keys/tokens from logs or environment.

## Known Dry-Run Notes

- The correct Codex CLI order is `codex -a never exec ...`, not
  `codex exec -a never ...`.
- For long prompts, write the prompt to `/tmp/codex_handoff_prompt.txt` on the
  server and run `codex -a never exec ... - < /tmp/codex_handoff_prompt.txt` to
  avoid shell word splitting.
- `failed to refresh available models: timeout waiting for child process to
  exit` may appear as a non-blocking remote Codex smoke warning if the requested
  task still completes. Record it and escalate only if it becomes frequent.
- A local Windows environment may lack `pytest`. If `py_compile` passes but
  `pytest` is missing, record that as a local environment gap rather than a code
  failure. Server or prepared local env can run pytest later.
- Stock preflight should require general token ignore rules so token-like local
  files do not enter Git accidentally.


# GitHub File Map

Updated: 2026-07-01

GitHub target: `https://github.com/cfy2yue/StockHome`

URL notes: URLs below use `main` as the intended publication branch. The local
`/data/cyx/1030/stock` directory is initialized as a Git worktree with origin
`https://github.com/cfy2yue/StockHome.git`. Remote `main` was initialized from
the server workspace on 2026-07-01.

| Local path | Git repo | GitHub URL | Purpose | Track in git? | Notes |
|---|---|---|---|---|---|
| `README.md` | `cfy2yue/StockHome` | `https://github.com/cfy2yue/StockHome/blob/main/README.md` | Project entrypoint | yes | Server/GitHub/Windows split. |
| `AGENTS.md` | `cfy2yue/StockHome` | `https://github.com/cfy2yue/StockHome/blob/main/AGENTS.md` | Stock agent rules | yes | User-facing Chinese protocol. |
| `goal.md` | `cfy2yue/StockHome` | `https://github.com/cfy2yue/StockHome/blob/main/goal.md` | Current product/research goal | yes | Top of file is current authority. |
| `.gitignore` | `cfy2yue/StockHome` | `https://github.com/cfy2yue/StockHome/blob/main/.gitignore` | Git exclusion policy | yes | Already excludes secrets/caches. |
| `docs/START_HERE.md` | `cfy2yue/StockHome` | `https://github.com/cfy2yue/StockHome/blob/main/docs/START_HERE.md` | Short onboarding | yes | Current entrypoint. |
| `docs/PROJECT_ENTRY.md` | `cfy2yue/StockHome` | `https://github.com/cfy2yue/StockHome/blob/main/docs/PROJECT_ENTRY.md` | Agent/engineer entry | yes | Keep concise. |
| `docs/GIT_AND_COLLABORATION.md` | `cfy2yue/StockHome` | `https://github.com/cfy2yue/StockHome/blob/main/docs/GIT_AND_COLLABORATION.md` | Git and ownership rules | yes | Created for collaboration. |
| `docs/GITHUB_FILE_MAP.md` | `cfy2yue/StockHome` | `https://github.com/cfy2yue/StockHome/blob/main/docs/GITHUB_FILE_MAP.md` | File URL map | yes | This file. |
| `docs/LOCAL_CODEX_DRY_RUN.md` | `cfy2yue/StockHome` | `https://github.com/cfy2yue/StockHome/blob/main/docs/LOCAL_CODEX_DRY_RUN.md` | Local Codex dry-run notes | yes | Supports workflow testing before normal CC flow. |
| `docs/USER_GUIDE.md` | `cfy2yue/StockHome` | `https://github.com/cfy2yue/StockHome/blob/main/docs/USER_GUIDE.md` | User guide | yes | User-facing docs. |
| `docs/RESPONSE_PROTOCOL.md` | `cfy2yue/StockHome` | `https://github.com/cfy2yue/StockHome/blob/main/docs/RESPONSE_PROTOCOL.md` | Answer protocol | yes | Must match `AGENTS.md`. |
| `docs/DATA_SOURCE_POLICY.md` | `cfy2yue/StockHome` | `https://github.com/cfy2yue/StockHome/blob/main/docs/DATA_SOURCE_POLICY.md` | Data-source and credential policy | yes | No tokens. |
| `src/`, `scripts/`, `tests/`, `config/`, `examples/` | `cfy2yue/StockHome` | `https://github.com/cfy2yue/StockHome/tree/main/src` | Code and small configs | yes | Verify no credentials before commit. |
| `reports/`, `data/`, `runs/`, `logs/`, `memory/` | none by default | not tracked | Large/local evidence and caches | no | Preserve locally; archive only with manifest and approval. |
| `docs/local_archive/` | none | not tracked | Full pre-slim Markdown history | no | Server-local provenance only. |
| `ds_api.txt`, `tushare_token.txt`, `.env*` | none | not tracked | Local credentials | no | Never print, commit, or quote. |

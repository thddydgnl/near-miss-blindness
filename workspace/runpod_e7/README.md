# runpod_e7 — B E7 (PointMaze) RunPod tooling

- Copy of `runpod_rss/` (itself a copy of `runpod_main/`) with pod prefix `bx7`, its own ledger, LIMIT $28 (user cap $30), stage cap `caps.json {"e7": 20}` (DESIGN_main §12.1).
- Usage is the same as `workspace/runpod_infra/PILOT_RULES.md` §2: `pod_up.sh <tag> "<gpu1>|<gpu2>" [cdisk] [vol]`, `rsh.sh <tag> '<cmd>'`, `sync.sh <tag> push|pull ...`, `pod_down.sh <tag>`, `budget.py status|watch 300`.
- Only pods recorded in `pods.ledger` are ours; never touch other prefixes.

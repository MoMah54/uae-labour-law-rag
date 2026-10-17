## Evaluation: hard_rerank_final (full pipeline)
_2026-10-17 08:58, questions_hard.jsonl, retrieval mode: rerank, 26 questions (21 in scope, 5 out of scope)_

| Retrieval | All | English | Arabic |
|---|---|---|---|
| Hit rate (top 6) | 100.0% | 100.0% | 100.0% |
| Correct article ranked first | 90.5% | 90.9% | 90.0% |
| MRR | 0.926 | 0.927 | 0.925 |
| Average search time | 12.41s | 11.39s | 13.54s |

| Answers | All | English | Arabic |
|---|---|---|---|
| Answer rate (in scope) | 95.2% | 90.9% | 100.0% |
| Citation accuracy | 95.2% | 90.9% | 100.0% |
| Citation precision | 100.0% | 100.0% | 100.0% |
| Refusal accuracy (out of scope) | 100.0% | 100.0% | 100.0% |
| Average latency | 14.8s | 12.6s | 17.3s |

Cross-lingual agreement (EN/AR pairs citing a common article): n/a
Questions that errored: 0

| Category | Questions | Retrieval hit rate | Citation accuracy | Refusal accuracy |
|---|---|---|---|---|
| near_scope | 5 | n/a | n/a | 100.0% |
| scenario | 17 | 100.0% | 100.0% | n/a |
| scope_rule | 2 | 100.0% | 100.0% | n/a |
| trap_no_number | 2 | 100.0% | 50.0% | n/a |

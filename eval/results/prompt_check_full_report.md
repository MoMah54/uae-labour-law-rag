## Evaluation: prompt_check (full pipeline)
_2026-10-17 09:11, questions_hard.jsonl, retrieval mode: rerank, 7 questions (2 in scope, 5 out of scope)_

| Retrieval | All | English | Arabic |
|---|---|---|---|
| Hit rate (top 6) | 100.0% | 100.0% | 100.0% |
| Correct article ranked first | 50.0% | 0.0% | 100.0% |
| MRR | 0.6 | 0.2 | 1.0 |
| Average search time | 18.19s | 21.74s | 14.64s |

| Answers | All | English | Arabic |
|---|---|---|---|
| Answer rate (in scope) | 100.0% | 100.0% | 100.0% |
| Citation accuracy | 100.0% | 100.0% | 100.0% |
| Citation precision | 100.0% | 100.0% | 100.0% |
| Refusal accuracy (out of scope) | 100.0% | 100.0% | 100.0% |
| Average latency | 17.0s | 16.2s | 18.1s |

Cross-lingual agreement (EN/AR pairs citing a common article): n/a
Questions that errored: 0

| Category | Questions | Retrieval hit rate | Citation accuracy | Refusal accuracy |
|---|---|---|---|---|
| near_scope | 5 | n/a | n/a | 100.0% |
| trap_no_number | 2 | 100.0% | 100.0% | n/a |

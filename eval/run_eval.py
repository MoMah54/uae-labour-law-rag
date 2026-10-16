"""
Phase 5: Evaluate the assistant on a bilingual test set.

Usage (from the project root, with .venv active and the database running):
    python eval/run_eval.py --retrieval-only --label baseline     # search only: free, fast, no Gemini
    python eval/run_eval.py --label baseline                      # full pipeline with Gemini (~15 min)
    python eval/run_eval.py --label baseline --resume eval/results/baseline_full.jsonl

Metrics
    Retrieval hit rate   an expected article is in the retrieved results (in-scope questions)
    MRR                  mean reciprocal rank of the first expected article (1.0 = always ranked first)
    Answer rate          in-scope questions the model answered instead of refusing
    Citation accuracy    answered questions whose citations include an expected article
    Citation precision   share of all citations that were expected articles
    Refusal accuracy     out-of-scope questions correctly refused
    Cross-lingual agreement   English/Arabic pairs that cite at least one common article
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate import answer_question, get_provider  # noqa: E402
from retrieve import retrieve  # noqa: E402

QUESTIONS_FILE = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"


def load_questions() -> list[dict]:
    with QUESTIONS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_hit_rank(retrieved_chunks: list[dict], expected: list[int]) -> int | None:
    """1-based rank of the first chunk belonging to an expected article."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk["article_number"] in expected:
            return rank
    return None


def pct(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 1) if denominator else None


def run_one(q: dict, provider, retrieval_only: bool) -> dict:
    record = {"id": q["id"], "pair_id": q["pair_id"], "language": q["language"],
              "in_scope": q["in_scope"], "expected_articles": q["expected_articles"],
              "question": q["question"]}
    chunks = retrieve(q["question"])
    record["retrieved_articles_ranked"] = [c["article_number"] for c in chunks]
    record["first_hit_rank"] = first_hit_rank(chunks, q["expected_articles"]) if q["in_scope"] else None

    if retrieval_only:
        return record
    try:
        out = answer_question(q["question"], provider=provider)
        record.update({"answer": out["answer"], "found": out["found"],
                       "cited_articles": out["cited_articles"], "provider": out["provider"],
                       "latency_seconds": out["latency_seconds"], "error": None})
    except Exception as error:  # keep going; one failed question shouldn't stop the run
        record.update({"answer": None, "found": None, "cited_articles": [], "error": str(error)[:200]})
    return record


def summarize(results: list[dict], retrieval_only: bool) -> dict:
    in_scope = [r for r in results if r["in_scope"]]
    summary = {"questions": len(results), "in_scope": len(in_scope),
               "out_of_scope": len(results) - len(in_scope)}

    def retrieval_metrics(rows):
        hits = [r for r in rows if r["first_hit_rank"] is not None]
        mrr = sum(1 / r["first_hit_rank"] for r in hits) / len(rows) if rows else None
        top1 = sum(1 for r in hits if r["first_hit_rank"] == 1)
        return {"hit_rate": pct(len(hits), len(rows)), "top1_rate": pct(top1, len(rows)),
                "mrr": round(mrr, 3) if mrr is not None else None}

    summary["retrieval"] = {"all": retrieval_metrics(in_scope)}
    for lang in ("en", "ar"):
        summary["retrieval"][lang] = retrieval_metrics([r for r in in_scope if r["language"] == lang])

    if retrieval_only:
        return summary

    ok = [r for r in results if r.get("error") is None]
    summary["errors"] = len(results) - len(ok)

    def answer_metrics(rows):
        ins = [r for r in rows if r["in_scope"]]
        outs = [r for r in rows if not r["in_scope"]]
        answered = [r for r in ins if r["found"]]
        correct = [r for r in answered if set(r["cited_articles"]) & set(r["expected_articles"])]
        all_cites = [(a, r["expected_articles"]) for r in answered for a in r["cited_articles"]]
        precise = sum(1 for a, exp in all_cites if a in exp)
        refused = [r for r in outs if r["found"] is False]
        latencies = [r["latency_seconds"] for r in rows if r.get("latency_seconds")]
        return {
            "answer_rate": pct(len(answered), len(ins)),
            "citation_accuracy": pct(len(correct), len(ins)),
            "citation_precision": pct(precise, len(all_cites)),
            "refusal_accuracy": pct(len(refused), len(outs)),
            "avg_latency_seconds": round(sum(latencies) / len(latencies), 1) if latencies else None,
        }

    summary["answers"] = {"all": answer_metrics(ok)}
    for lang in ("en", "ar"):
        summary["answers"][lang] = answer_metrics([r for r in ok if r["language"] == lang])

    # Cross-lingual agreement: do the English and Arabic versions cite a common article?
    by_pair: dict[str, dict] = {}
    for r in ok:
        if r["pair_id"]:
            by_pair.setdefault(r["pair_id"], {})[r["language"]] = r
    complete = [p for p in by_pair.values() if "en" in p and "ar" in p]
    agree = sum(1 for p in complete if set(p["en"]["cited_articles"]) & set(p["ar"]["cited_articles"]))
    summary["cross_lingual_agreement"] = pct(agree, len(complete))
    return summary


def to_markdown(summary: dict, label: str, retrieval_only: bool) -> str:
    def cell(v, unit="%"):
        return "n/a" if v is None else f"{v}{unit}"

    lines = [f"## Evaluation: {label} ({'retrieval only' if retrieval_only else 'full pipeline'})",
             f"_{datetime.now():%Y-%m-%d %H:%M}, {summary['questions']} questions "
             f"({summary['in_scope']} in scope, {summary['out_of_scope']} out of scope)_", "",
             "| Retrieval | All | English | Arabic |", "|---|---|---|---|"]
    r = summary["retrieval"]
    lines.append(f"| Hit rate (top 6) | {cell(r['all']['hit_rate'])} | {cell(r['en']['hit_rate'])} | {cell(r['ar']['hit_rate'])} |")
    lines.append(f"| Correct article ranked first | {cell(r['all']['top1_rate'])} | {cell(r['en']['top1_rate'])} | {cell(r['ar']['top1_rate'])} |")
    lines.append(f"| MRR | {cell(r['all']['mrr'], '')} | {cell(r['en']['mrr'], '')} | {cell(r['ar']['mrr'], '')} |")
    if not retrieval_only:
        a = summary["answers"]
        lines += ["", "| Answers | All | English | Arabic |", "|---|---|---|---|"]
        for key, name, unit in [("answer_rate", "Answer rate (in scope)", "%"),
                                ("citation_accuracy", "Citation accuracy", "%"),
                                ("citation_precision", "Citation precision", "%"),
                                ("refusal_accuracy", "Refusal accuracy (out of scope)", "%"),
                                ("avg_latency_seconds", "Average latency", "s")]:
            lines.append(f"| {name} | {cell(a['all'][key], unit)} | {cell(a['en'][key], unit)} | {cell(a['ar'][key], unit)} |")
        lines += ["", f"Cross-lingual agreement (EN/AR pairs citing a common article): "
                      f"{cell(summary['cross_lingual_agreement'])}",
                  f"Questions that errored: {summary['errors']}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="name for this run, e.g. baseline or repaired")
    parser.add_argument("--retrieval-only", action="store_true", help="skip Gemini; measure search only")
    parser.add_argument("--pause", type=float, default=13.0,
                        help="seconds between Gemini calls (free tier allows about 5 per minute)")
    parser.add_argument("--resume", help="results .jsonl from an interrupted run; finished questions are skipped")
    parser.add_argument("--limit", type=int, help="only run the first N questions (for a quick test)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mode = "retrieval" if args.retrieval_only else "full"
    results_path = Path(args.resume) if args.resume else RESULTS_DIR / f"{args.label}_{mode}.jsonl"

    done: dict[str, dict] = {}
    if args.resume and results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("error") is None:
                    done[r["id"]] = r
        print(f"Resuming: {len(done)} questions already done")
    elif results_path.exists():
        results_path.unlink()

    questions = load_questions()[: args.limit] if args.limit else load_questions()
    provider = None if args.retrieval_only else get_provider()

    results = list(done.values())
    todo = [q for q in questions if q["id"] not in done]
    for i, q in enumerate(todo, start=1):
        record = run_one(q, provider, args.retrieval_only)
        results.append(record)
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        status = f"rank {record['first_hit_rank']}" if q["in_scope"] else "out of scope"
        if not args.retrieval_only:
            status += " | ERROR" if record.get("error") else f" | cited {record['cited_articles']}"
        print(f"[{i}/{len(todo)}] {q['id']:<28} expected {q['expected_articles']} | {status}")

        if not args.retrieval_only and i < len(todo):
            time.sleep(args.pause)

    summary = summarize(results, args.retrieval_only)
    report = to_markdown(summary, args.label, args.retrieval_only)
    (RESULTS_DIR / f"{args.label}_{mode}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR / f"{args.label}_{mode}_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
"""EXP-R01 step 3 and EXP-R02 (env mp1): score the judge cache and compare with our system.

REF-R01a  chosen_index as returned.
REF-R01b  candidates whose parsed answer equals own_answer (highest judge confidence among
          ties); chosen_index when none matches.
Both are written in the official prediction format and scored with score_predictions.py.
EXP-R02   val_hard 2 x 2 table against EXP-B12 arm A seed 42 + POST-P01/P02, and the list of
          questions both get wrong (input to the manual residual review).
Reference outputs stay in dev/runs/ref and are never read by submission code.
"""
import json
import sys
from pathlib import Path

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

import numpy as np  # noqa: E402

from experiments.exp_b07_cross_attn import official_score, write_official_predictions  # noqa: E402
from experiments.multiseed import RUNS, SplitData, rel  # noqa: E402

import common  # noqa: E402

REF_RUNS = RUNS / "ref"
CACHE = REF_RUNS / "exp_r01_judge_cache_openai_gpt-6.1-sol_effort-high.jsonl"
PILOTS = {
    "openai_gpt-6.1-sol_default": REF_RUNS / "exp_r01_judge_cache_openai_gpt-6.1-sol.jsonl",
    "gemini_gemini-3.8-flash_free": REF_RUNS / "exp_r01_judge_cache_gemini_gemini-3.8-flash.jsonl",
}
OURS = RUNS / "predictions" / "EXP-B12_embeddings_only_seed42_post_predictions.json"
REVIEW_LIMIT = 30


def load_cache(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return {entry["id"]: entry for entry in (json.loads(line) for line in handle if line.strip())}


def parse_own(value) -> str | None:
    return common.parse_answer(f"#### {value}") if value is not None else None


def decisions(val: SplitData, cache: dict):
    n = len(val.bank)
    r01a = np.zeros(n, dtype=int)
    r01b = np.zeros(n, dtype=int)
    own = [None] * n
    failed = []
    for i, row in enumerate(val.bank):
        entry = cache[row["id"]]
        if entry["failed"]:
            failed.append(row["id"])
            wrong = np.flatnonzero(val.labels[i] == 0)
            r01a[i] = r01b[i] = int(wrong[0]) if len(wrong) else 0
            continue
        parsed = entry["parsed"]
        r01a[i] = int(parsed["chosen_index"])
        own[i] = parse_own(parsed.get("own_answer"))
        confidence = {c.get("index"): float(c.get("confidence") or 0.0) for c in parsed.get("candidates", [])}
        matches = [c["index"] for c in row["candidates"] if own[i] is not None and common.parse_answer(c["text"]) == own[i]]
        r01b[i] = max(matches, key=lambda k: (confidence.get(k, 0.0), -k)) if matches else r01a[i]
    return r01a, r01b, own, failed


def hit_counts(val: SplitData, choice: np.ndarray) -> dict:
    hits = val.labels[np.arange(len(val.bank)), choice].astype(bool)
    return {"val_hard": int(hits[val.hard].sum()), "val_all": int(hits.sum()), "_hits": hits}


def main() -> None:
    val = SplitData("validation")
    gold = [common.parse_answer(g["answer"]) for g in common.read_jsonl(common.DATA / "gsm8k_validation.jsonl")]
    cache = load_cache(CACHE)
    if len(cache) != len(val.bank):
        raise RuntimeError(f"cache has {len(cache)} of {len(val.bank)} questions")
    r01a, r01b, own, failed = decisions(val, cache)

    scores = {}
    for name, choice in (("R01a", r01a), ("R01b", r01b)):
        pred = REF_RUNS / f"exp_r01_{name}_predictions.json"
        write_official_predictions(pred, val.bank, choice)
        official = official_score(pred, REF_RUNS / f"exp_r01_{name}_score.json")
        counts = hit_counts(val, choice)
        if int(official["correct"]) != counts["val_all"]:
            raise RuntimeError(f"{name}: official {official['correct']} != local {counts['val_all']}")
        scores[name] = {"val_hard": counts["val_hard"], "val_all": counts["val_all"], "official": official,
                        "predictions": rel(pred), "_hits": counts["_hits"]}
        print(f"{name}: val_hard {counts['val_hard']}/355 val_all {counts['val_all']}/798")

    own_ok = np.array([o is not None and o == g for o, g in zip(own, gold)])
    r01a_hits = scores["R01a"]["_hits"]
    n_correct = val.labels.sum(axis=1)
    by_k = {}
    for k in (1, 2, 3):
        mask = val.hard & (n_correct == k)
        by_k[str(k)] = {"questions": int(mask.sum()), "R01a_correct": int(r01a_hits[mask].sum()),
                        "R01b_correct": int(scores["R01b"]["_hits"][mask].sum())}
    diagnostics = {
        "own_answer_correct": {"val_all": int(own_ok.sum()), "val_hard": int(own_ok[val.hard].sum())},
        "own_answer_unparsed": int(sum(o is None for o in own)),
        "R01a_wrong_but_own_correct": {"val_all": int((~r01a_hits & own_ok).sum()),
                                       "val_hard": int((~r01a_hits & own_ok)[val.hard].sum())},
        "R01a_wrong_and_own_wrong": {"val_all": int((~r01a_hits & ~own_ok).sum()),
                                     "val_hard": int((~r01a_hits & ~own_ok)[val.hard].sum())},
        "R01b_differs_from_R01a": int((r01a != r01b).sum()),
        "val_hard_by_correct_candidates": by_k,
        "failed_questions": failed,
        "retried_questions": sum(len(e["attempts"]) > 1 for e in cache.values()),
    }

    entries = list(cache.values())
    usage = {key: int(sum(e["usage_total"][key] for e in entries)) for key in ("prompt", "candidates", "thoughts")}
    run = {
        "backend": "openai", "endpoint": entries[0]["endpoint"], "model": entries[0]["model"],
        "model_versions": sorted({a["model_version"] for e in entries for a in e["attempts"]}),
        "reasoning_effort": "high", "temperature": "model default", "tier": "standard (real-time)",
        "price_per_million_usd": entries[0]["price_per_million"],
        "called_between": [min(e["called_at"] for e in entries), max(e["called_at"] for e in entries)],
        "calls": int(sum(len(e["attempts"]) for e in entries)), "usage_tokens": usage,
        "cost_usd": round(sum(e["cost_usd"] for e in entries), 4),
        "mean_seconds_per_question": float(np.mean([e["seconds"] for e in entries])),
        "cache": rel(CACHE), "prompt_file": "dev/src/reference/prompts/judge_v1.txt",
    }

    pilots = {}
    for name, path in PILOTS.items():
        if not path.exists():
            continue
        pilot = load_cache(path)
        idx = [i for i, row in enumerate(val.bank) if row["id"] in pilot]
        hits_pilot = sum(int(val.labels[i, pilot[val.bank[i]["id"]]["parsed"]["chosen_index"]]) for i in idx
                         if not pilot[val.bank[i]["id"]]["failed"])
        pilots[name] = {"questions": len(idx), "R01a_correct": hits_pilot,
                        "high_effort_correct_same_questions": int(r01a_hits[idx].sum()),
                        "cost_usd": round(sum(e["cost_usd"] for e in pilot.values()), 4), "cache": rel(path)}

    ours = json.loads(OURS.read_text(encoding="utf-8"))
    if ours["ids"] != [row["id"] for row in val.bank]:
        raise RuntimeError("our predictions are not aligned with the validation bank")
    ours_hits = val.labels[np.arange(len(val.bank)), np.asarray(ours["selected"])].astype(bool)
    h = val.hard
    table = {
        "both_correct": int((r01a_hits & ours_hits)[h].sum()),
        "only_judge_correct": int((r01a_hits & ~ours_hits)[h].sum()),
        "only_ours_correct": int((~r01a_hits & ours_hits)[h].sum()),
        "both_wrong": int((~r01a_hits & ~ours_hits)[h].sum()),
    }
    both_wrong_idx = [i for i in np.flatnonzero(h & ~r01a_hits & ~ours_hits)]
    review = []
    for i in both_wrong_idx[:REVIEW_LIMIT]:
        row, parsed = val.bank[i], cache[val.bank[i]["id"]]["parsed"]
        review.append({
            "id": row["id"], "question": row["question"], "reference_answer": gold[i],
            "candidate_answers": [common.parse_answer(c["text"]) for c in row["candidates"]],
            "correct_candidates": [int(k) for k in np.flatnonzero(val.labels[i])],
            "judge_choice": int(r01a[i]), "judge_own_answer": parsed.get("own_answer"),
            "ours_choice": int(ours["selected"][i]),
            "judge_first_errors": {str(c.get("index")): c.get("first_error") for c in parsed.get("candidates", [])},
            "review": None,
        })
    review_path = REF_RUNS / "residual_review.json"
    review_path.write_text(json.dumps({"note": "report analysis only; never used for training or selection",
                                       "both_wrong_val_hard": len(both_wrong_idx), "listed": len(review),
                                       "items": review}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"R02 val_hard: {table}")
    print(f"diagnostics: {json.dumps({k: v for k, v in diagnostics.items() if k != 'failed_questions'})}")
    print(f"pilots: {pilots}")
    print(f"run: cost ${run['cost_usd']} tokens {usage} versions {run['model_versions']}")

    for s in scores.values():
        s.pop("_hits")
    audit_r01 = {"experiment": "EXP-R01", "reference_only": True, "run": run, "scores": scores,
                 "diagnostics": diagnostics, "pilots": pilots,
                 "ref_r00": {"oracle": {"val_all": 798, "val_hard": 355},
                             "random_expectation": {"val_all": float(n_correct.sum() / 4),
                                                    "val_hard": float(n_correct[h].sum() / 4)}}}
    audit_r02 = {"experiment": "EXP-R02", "reference_only": True, "judge": "REF-R01a", "ours": rel(OURS),
                 "ours_counts": {"val_hard": int(ours_hits[h].sum()), "val_all": int(ours_hits.sum())},
                 "val_hard_2x2": table, "residual_review": rel(review_path)}
    logs = RUNS / "audit_logs"
    (logs / "exp_r01_judge_audit.json").write_text(json.dumps(audit_r01, indent=2, ensure_ascii=False), encoding="utf-8")
    (logs / "exp_r02_headroom_audit.json").write_text(json.dumps(audit_r02, indent=2, ensure_ascii=False), encoding="utf-8")
    print("audits written")


if __name__ == "__main__":
    main()

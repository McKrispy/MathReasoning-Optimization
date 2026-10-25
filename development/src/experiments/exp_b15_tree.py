"""EXP-B15: tree reranker supplement (MOD-M04a / MOD-M04b, FEAT-G04b, POST-P03).

R0    HistGradientBoosting on train_hard, seed 42: environment check against EXP-B10.
Grid  {M04a classifier, M04b LightGBM lambdarank} x {train_hard, train_all} x seeds 42/43/44.
R5    best grid configuration + FEAT-G04b (TruncatedSVD 64 + within-question centered 64).
Gate  best tree vs the EXP-B12 arm A single model + POST-P01/P02, eight EXP-B10 gates per seed.

Columns are the B04 locked 16 (G01 core + G03) from the B02 contract. Tree
hyperparameters are not searched. Early stopping is off for every tree so the
train_all rows (19,020 > 10,000) do not silently switch it on. Test
references are not opened.
"""
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

from experiments.multiseed import (  # noqa: E402
    RUNS, SEEDS, SplitData, decode, environment_record, reasoning_struct, register_seeds, rel, retention,
    save_npz, summarize, configure_determinism,
)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from experiments.exp_b07_cross_attn import official_score, write_official_predictions  # noqa: E402
from experiments.exp_b10_fusion import gate_specs  # noqa: E402
from postprocessing.aggregation import apply_disagreement_gate  # noqa: E402
from postprocessing.calibration import TemperatureScaler  # noqa: E402

HGB_PARAMS = {"max_iter": 100, "max_depth": 5, "min_samples_leaf": 20, "l2_regularization": 1.0,
              "early_stopping": False}
LGB_PARAMS = {"objective": "lambdarank", "n_estimators": 100, "max_depth": 5, "num_leaves": 31,
              "min_child_samples": 20, "reg_lambda": 1.0, "learning_rate": 0.1, "deterministic": True,
              "force_row_wise": True, "num_threads": 4, "verbose": -1}
B10_TREE = {"val_hard": 261, "val_all": 704}


def svd_features(train: SplitData, val: SplitData, seed: int = 42):
    svd = TruncatedSVD(n_components=64, random_state=seed)
    flat = train.emb.reshape(-1, train.emb.shape[-1])
    svd.fit(flat)

    def transform(split: SplitData) -> np.ndarray:
        proj = svd.transform(split.emb.reshape(-1, split.emb.shape[-1])).reshape(len(split.bank), 4, 64)
        centered = proj - proj.mean(axis=1, keepdims=True)
        return np.concatenate([proj, centered], axis=-1).astype(np.float32)

    return transform(train), transform(val), svd


def fit_tree(kind: str, x: np.ndarray, y: np.ndarray, seed: int):
    n, c, d = x.shape
    flat_x, flat_y = x.reshape(n * c, d), y.reshape(-1)
    tick = time.perf_counter()
    if kind == "M04a":
        model = HistGradientBoostingClassifier(random_state=seed, **HGB_PARAMS).fit(flat_x, flat_y)
    else:
        model = lgb.LGBMRanker(random_state=seed, **LGB_PARAMS)
        model.fit(flat_x, flat_y.astype(int), group=[c] * n)
    return model, time.perf_counter() - tick


def tree_logits(kind: str, model, x: np.ndarray) -> np.ndarray:
    n, c, d = x.shape
    flat = x.reshape(n * c, d)
    raw = model.decision_function(flat) if kind == "M04a" else model.predict(flat)
    return np.asarray(raw, dtype=np.float64).reshape(n, c)


def model_bytes(kind: str, model, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "M04a":
        with path.open("wb") as handle:
            pickle.dump(model, handle)
    else:
        model.booster_.save_model(str(path))
    return path.stat().st_size


def run_config(name: str, kind: str, rows: str, x_train, x_val, train: SplitData, val: SplitData,
               out_dir: Path, registry: List[Dict[str, Any]]) -> Dict[str, Any]:
    fit_mask = train.hard if rows == "train_hard" else np.ones(len(train.bank), dtype=bool)
    per_seed, logits = {}, {}
    for seed in SEEDS:
        model, fit_seconds = fit_tree(kind, x_train[fit_mask], train.labels[fit_mask], seed)
        tick = time.perf_counter()
        val_logits = tree_logits(kind, model, x_val)
        infer_seconds = time.perf_counter() - tick
        suffix = "pkl" if kind == "M04a" else "txt"
        path = out_dir / f"seed_{seed}" / f"{name}.{suffix}"
        size = model_bytes(kind, model, path)
        import hashlib
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        registry.append({"arm": name, "seed": seed, "member": "tree", "derived": {"random_state": seed},
                         "file": rel(path), "sha256": digest})
        decoded = decode(val_logits, val)
        per_seed[seed] = {"decoded": decoded, "fit_seconds": fit_seconds, "infer_seconds_val_all": infer_seconds,
                          "bytes": size, "sha256": digest, "file": rel(path)}
        logits[f"seed_{seed}"] = val_logits
    np.savez(out_dir / f"val_logits_{name}.npz", **logits)
    series = lambda key, split: [per_seed[s]["decoded"][key][split]["correct"] for s in SEEDS]  # noqa: E731
    result = {
        "kind": kind,
        "fit_rows": rows,
        "input_dim": int(x_train.shape[-1]),
        "per_seed": {str(s): {**{k: v for k, v in per_seed[s].items() if k != "decoded"},
                              "decoded": {k: v for k, v in per_seed[s]["decoded"].items()
                                          if k not in ("argmax_choice", "post_choice")}} for s in SEEDS},
        "summary": {
            "argmax_val_hard": summarize(series("argmax", "val_hard")),
            "argmax_val_all": summarize(series("argmax", "val_all")),
            "post_val_hard": summarize(series("post", "val_hard")),
            "post_val_all": summarize(series("post", "val_all")),
        },
        "seed_sensitive": len({per_seed[s]["sha256"] for s in SEEDS}) > 1,
        "validation_configs_evaluated": sum(per_seed[s]["decoded"]["validation_configs_evaluated"] for s in SEEDS),
    }
    result["_runtime"] = {s: {"logits": logits[f"seed_{s}"], "decoded": per_seed[s]["decoded"]} for s in SEEDS}
    s = result["summary"]
    print(f"  {name}: argmax val_hard {s['argmax_val_hard']['values']} | post val_hard {s['post_val_hard']['values']} "
          f"val_all {s['post_val_all']['values']} | seed-sensitive {result['seed_sensitive']}")
    return result


def run_gates(tree: Dict[str, Any], val: SplitData) -> Dict[str, Any]:
    b12 = np.load(RUNS / "exp_b12" / "val_logits_embeddings_only.npz")
    per_seed_rows = {}
    for seed in SEEDS:
        m03_logits = b12[f"seed_{seed}_full"]
        m03 = decode(m03_logits, val)
        m03_probs = TemperatureScaler(m03["post"]["temperature"]).transform(m03_logits)
        m03_choice = m03["post_choice"]
        t = tree["_runtime"][seed]["decoded"]
        t_probs = TemperatureScaler(t["post"]["temperature"]).transform(tree["_runtime"][seed]["logits"])
        t_choice = t["post_choice"]
        idx = np.arange(len(val.bank))
        rows = []
        for spec in gate_specs():
            chosen = apply_disagreement_gate(m03_choice, t_choice, m03_probs[idx, m03_choice],
                                             t_probs[idx, t_choice], spec["name"], spec["threshold"])
            hits = val.labels[idx, chosen].astype(bool)
            rows.append({"name": spec["name"], "threshold": spec["threshold"], "priority": spec["priority"],
                         "val_hard": int(hits[val.hard].sum()), "val_all": int(hits.sum())})
        disagree = (m03_choice != t_choice) & val.hard
        m03_hits = val.labels[idx, m03_choice].astype(bool)
        t_hits = val.labels[idx, t_choice].astype(bool)
        per_seed_rows[str(seed)] = {
            "gates": rows,
            "val_hard_disagreements": int(disagree.sum()),
            "only_m03": int((m03_hits & ~t_hits)[disagree].sum()),
            "only_tree": int((~m03_hits & t_hits)[disagree].sum()),
        }
    names = [(r["name"], r["threshold"], r["priority"]) for r in per_seed_rows[str(SEEDS[0])]["gates"]]
    mean_rows = []
    for i, (name, threshold, priority) in enumerate(names):
        hard = [per_seed_rows[str(s)]["gates"][i]["val_hard"] for s in SEEDS]
        all_ = [per_seed_rows[str(s)]["gates"][i]["val_all"] for s in SEEDS]
        mean_rows.append({"name": name, "threshold": threshold, "priority": priority,
                          "val_hard": summarize(hard), "val_all": summarize(all_)})
    m03_all = mean_rows[0]["val_all"]["mean"]
    eligible = [r for r in mean_rows if r["val_all"]["mean"] >= m03_all] or mean_rows
    selected = max(eligible, key=lambda r: (r["val_hard"]["mean"], r["val_all"]["mean"], -r["priority"]))
    return {"per_seed": per_seed_rows, "mean_over_seeds": mean_rows,
            "selected": {"name": selected["name"], "threshold": selected["threshold"]},
            "rule": "max 3-seed mean val_hard among gates with mean val_all >= m03_only; ties higher val_all then m03_only"}


def run_exp_b15() -> Dict[str, Any]:
    print("=" * 70)
    print("Executing EXP-B15: tree supplement")
    print("=" * 70)
    device = configure_determinism()
    env = environment_record(device)
    train = SplitData("train")
    val = SplitData("validation")
    struct_train, g01_columns, g03_columns = reasoning_struct(train)
    struct_val, _, _ = reasoning_struct(val)
    out_dir = RUNS / "exp_b15"
    registry: List[Dict[str, Any]] = []

    print("\n--- R0: EXP-B10 tree in mp1 ---")
    r0_model, _ = fit_tree("M04a", struct_train[train.hard], train.labels[train.hard], 42)
    r0_choice = np.argmax(tree_logits("M04a", r0_model, struct_val), axis=1)
    r0_hits = val.labels[np.arange(len(val.bank)), r0_choice].astype(bool)
    r0 = {"val_hard": int(r0_hits[val.hard].sum()), "val_all": int(r0_hits.sum()), "b10": B10_TREE}
    r0["matches_b10"] = r0["val_hard"] == B10_TREE["val_hard"] and r0["val_all"] == B10_TREE["val_all"]
    print(f"  R0 {r0}")

    print("\n--- 2 x 2 grid ---")
    grid: Dict[str, Dict[str, Any]] = {}
    for kind in ("M04a", "M04b"):
        for rows in ("train_hard", "train_all"):
            name = f"{kind}_{rows}"
            grid[name] = run_config(name, kind, rows, struct_train, struct_val, train, val, out_dir, registry)
    ref = grid["M04a_train_hard"]["summary"]
    for name, res in grid.items():
        if name == "M04a_train_hard":
            continue
        res["retention_vs_M04a_train_hard"] = retention(
            res["summary"]["post_val_hard"]["values"], ref["post_val_hard"]["values"],
            res["summary"]["post_val_all"]["values"], ref["post_val_all"]["values"], paired=True)

    best = max(grid, key=lambda n: (grid[n]["summary"]["post_val_hard"]["mean"], grid[n]["summary"]["post_val_all"]["mean"]))
    print(f"\n--- R5: {best} + FEAT-G04b ---")
    svd_train, svd_val, svd = svd_features(train, val)
    svd_path = out_dir / "feat_g04b_svd.npz"
    svd_sha = save_npz(svd_path, {"components": svd.components_.astype(np.float32)})
    x5_train = np.concatenate([struct_train, svd_train], axis=-1)
    x5_val = np.concatenate([struct_val, svd_val], axis=-1)
    kind, rows = best.split("_", 1)
    r5 = run_config(f"{best}_plus_svd", kind, rows, x5_train, x5_val, train, val, out_dir, registry)
    r5["retention_vs_best_grid"] = retention(
        r5["summary"]["post_val_hard"]["values"], grid[best]["summary"]["post_val_hard"]["values"],
        r5["summary"]["post_val_all"]["values"], grid[best]["summary"]["post_val_all"]["values"], paired=True)
    r5["svd"] = {"file": rel(svd_path), "sha256": svd_sha, "bytes": int(svd.components_.astype(np.float32).nbytes),
                 "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()), "fit_on": "train_all"}

    candidates = {**grid, f"{best}_plus_svd": r5}
    gate_tree_name = max(candidates, key=lambda n: (candidates[n]["summary"]["post_val_hard"]["mean"],
                                                    candidates[n]["summary"]["post_val_all"]["mean"]))
    print(f"\n--- gates: {gate_tree_name} vs EXP-B12 arm A ---")
    gates = run_gates(candidates[gate_tree_name], val)
    for row in gates["mean_over_seeds"]:
        print(f"  {row['name']} t={row['threshold']}: val_hard {row['val_hard']['values']} val_all {row['val_all']['values']}")
    print(f"  selected {gates['selected']}")

    sel = gates["selected"]
    sel_rows = [r for r in gates["per_seed"]["42"]["gates"] if r["name"] == sel["name"] and r["threshold"] == sel["threshold"]]
    best_name_pred = RUNS / "predictions" / f"EXP-B15_{gate_tree_name}_seed42_post_predictions.json"
    write_official_predictions(best_name_pred, val.bank, candidates[gate_tree_name]["_runtime"][42]["decoded"]["post_choice"])
    official = official_score(best_name_pred, RUNS / "scores" / f"EXP-B15_{gate_tree_name}_seed42_post_score.json")

    for res in candidates.values():
        res.pop("_runtime", None)
    register_seeds("EXP-B15", registry)
    audit = {
        "experiment": "EXP-B15",
        "environment": env,
        "hgb_params": HGB_PARAMS,
        "lgb_params": LGB_PARAMS,
        "columns": list(g01_columns) + list(g03_columns),
        "seeds": list(SEEDS),
        "r0": r0,
        "grid": grid,
        "best_grid": best,
        "r5": r5,
        "gate_tree": gate_tree_name,
        "gates": gates,
        "selected_gate_seed42": sel_rows,
        "official_gate_tree_seed42_post": official,
        "validation_configs_evaluated": sum(r["validation_configs_evaluated"] for r in candidates.values())
        + len(gate_specs()) * len(SEEDS),
        "seed_registry": rel(RUNS / "audit_logs" / "seed_registry.json"),
        "note": "submission.py and model_bundle untouched; test references not opened",
    }
    path = RUNS / "audit_logs" / "exp_b15_tree_audit.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audit saved to: {path}")
    return audit


if __name__ == "__main__":
    run_exp_b15()

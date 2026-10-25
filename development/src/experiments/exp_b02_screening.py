"""EXP-B02: Feature Collinearity, Within-Problem Constant & Variance Screening.

Analyzes the full 56-feature matrix on train_hard (2,114 questions * 4 candidates = 8,456 rows).
Calculates:
1. Global standard deviation, range, non-zero ratio.
2. Within-problem standard deviation (detecting question-level constants where std == 0).
3. Near-constant features (std relative to range negligible or trigger rate < 1%).
4. Pearson correlation matrix and collinearity clustering (|r| >= 0.90).
5. Recommended feature representatives for linear models (EXP-B03).
"""
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple
import numpy as np

# Ensure dev/src is in sys.path
DEV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

from common_utils import (
    load_dataset,
    load_gold_references,
    get_hard_subset_mask,
    build_label_matrix,
)
from features.consensus import ConsensusFeatureExtractor
from features.surface import SurfaceFeatureExtractor
from features.symbolic import SymbolicVerificationFeatureExtractor
from features.geometry import GeometryFeatureExtractor


def derive_column_contract(results: Dict[str, Any]) -> Dict[str, Any]:
    """Columns later experiments may read. Derived only from screening fields.

    A column is statistically eligible when it is not a question-level constant,
    not a near-constant, and not a non-representative member of a |r| >= 0.90
    cluster. A representative that is itself near-constant contributes no column.
    """
    question_constants = [c["name"] for c in results["question_level_constants"]]
    near_constants = [c["name"] for c in results["near_constants"]]
    question_constant_set = set(question_constants)
    near_constant_set = set(near_constants)
    non_representatives = []
    suppressed_representatives = []
    for cluster in results["collinear_clusters"]:
        representative = cluster["representative"]
        for name in cluster["cluster_features"]:
            if name != representative:
                non_representatives.append(name)
        if representative in near_constant_set or representative in question_constant_set:
            suppressed_representatives.append(representative)
    non_representative_set = set(non_representatives)

    eligible = []
    for column in results["column_stats"]:
        name = column["name"]
        if name in question_constant_set or name in near_constant_set or name in non_representative_set:
            continue
        eligible.append(name)

    by_family: Dict[str, List[str]] = defaultdict(list)
    family_of = {column["name"]: column["family"] for column in results["column_stats"]}
    for name in eligible:
        by_family[family_of[name]].append(name)

    # Policy lists. They are not findings of the variance screen.
    # is_negative is also a near-constant; the other four morphology columns are not.
    gsm8k_morphology_columns_not_core = [
        "log_magnitude",
        "is_integer",
        "is_negative",
        "is_zero",
        "appears_in_question",
    ]
    contingency_table_columns = [
        "is_negative",
        "is_zero",
        "is_integer",
        "has_missing_condition",
    ]

    return {
        "source_audit": "dev/runs/audit_logs/exp_b02_screening_train_hard.json",
        "dataset": results["dataset"],
        "num_questions": results["num_questions"],
        "num_candidate_rows": results["num_candidate_rows"],
        "rule": (
            "Eligible iff not a question-level constant, not a near-constant, "
            "and not a non-representative of a |r|>=0.90 cluster. "
            "A near-constant representative contributes no column."
        ),
        "question_level_constants": question_constants,
        "near_constants": near_constants,
        "collinear_non_representatives": non_representatives,
        "near_constant_representatives_suppressed": suppressed_representatives,
        "statistically_eligible_columns": eligible,
        "statistically_eligible_count": len(eligible),
        "statistically_eligible_by_family": dict(by_family),
        "gsm8k_morphology_columns_not_core": gsm8k_morphology_columns_not_core,
        "contingency_table_columns": contingency_table_columns,
    }


def run_exp_b02_screening() -> Dict[str, Any]:
    print("=" * 70)
    print("Executing EXP-B02: Feature Screening on train_hard (2,114 questions)")
    print("=" * 70)

    # 1. Load dataset & extract train_hard
    bank_train, arrays_train = load_dataset('train')
    gold_train = load_gold_references('train')
    hard_mask = get_hard_subset_mask(bank_train)

    bank_hard = [b for i, b in enumerate(bank_train) if hard_mask[i]]
    arrays_hard = {'embeddings': arrays_train['embeddings'][hard_mask]}
    labels_hard = build_label_matrix(bank_hard, [g for i, g in enumerate(gold_train) if hard_mask[i]])

    n_q = len(bank_hard)
    n_rows = n_q * 4
    print(f"Loaded train_hard: {n_q} questions, {n_rows} candidate rows.")
    assert n_q == 2114, f"Expected 2114 questions in train_hard, got {n_q}"

    # 2. Extract all 56 features
    ext_g01 = ConsensusFeatureExtractor()
    ext_g02 = SurfaceFeatureExtractor()
    ext_g03 = SymbolicVerificationFeatureExtractor()
    ext_g04 = GeometryFeatureExtractor()

    print("Extracting FEAT-G01 (Consensus)...")
    x_g01 = ext_g01.transform(bank_hard, arrays_hard)  # [N, 4, 11]
    print("Extracting FEAT-G02 (Surface)...")
    x_g02 = ext_g02.transform(bank_hard, arrays_hard)  # [N, 4, 16]
    print("Extracting FEAT-G03 (Symbolic)...")
    x_g03 = ext_g03.transform(bank_hard, arrays_hard)  # [N, 4, 14]
    print("Extracting FEAT-G04 (Geometry)...")
    x_g04 = ext_g04.transform(bank_hard, arrays_hard)  # [N, 4, 15]

    x_all = np.concatenate([x_g01, x_g02, x_g03, x_g04], axis=-1).astype(np.float32)
    feature_names = (
        ext_g01.FEATURE_NAMES +
        ext_g02.FEATURE_NAMES +
        ext_g03.FEATURE_NAMES +
        ext_g04.FEATURE_NAMES
    )
    feature_families = (
        ['FEAT-G01'] * len(ext_g01.FEATURE_NAMES) +
        ['FEAT-G02'] * len(ext_g02.FEATURE_NAMES) +
        ['FEAT-G03'] * len(ext_g03.FEATURE_NAMES) +
        ['FEAT-G04'] * len(ext_g04.FEATURE_NAMES)
    )
    total_dim = len(feature_names)
    assert total_dim == 56, f"Expected 56 features, got {total_dim}"
    print(f"Total extracted feature tensor shape: {x_all.shape}")

    # 3. Candidate-level flattened array
    x_flat = x_all.reshape(n_rows, total_dim)

    # 4. Variance & Constant Analysis
    column_stats = []
    question_level_constants = []
    near_constants = []

    for j, (name, family) in enumerate(zip(feature_names, feature_families)):
        col_vals = x_flat[:, j]
        g_mean = float(np.mean(col_vals))
        g_std = float(np.std(col_vals))
        g_min = float(np.min(col_vals))
        g_max = float(np.max(col_vals))
        nonzero_ratio = float(np.count_nonzero(col_vals) / n_rows)

        # Within-problem standard deviation (across 4 candidates per question)
        within_stds = np.std(x_all[:, :, j], axis=1)  # shape [N]
        mean_within_std = float(np.mean(within_stds))
        max_within_std = float(np.max(within_stds))

        # Check if question-level constant (std == 0 across all questions)
        is_q_constant = (max_within_std < 1e-7)

        # Check if near constant (std < 0.01 or non-zero ratio < 1%)
        is_near_constant = False
        near_constant_reason = ""
        if is_q_constant:
            question_level_constants.append({
                "index": j,
                "name": name,
                "family": family,
                "global_mean": g_mean,
                "global_std": g_std,
            })
        else:
            val_range = g_max - g_min
            if g_std < 0.01:
                is_near_constant = True
                near_constant_reason = f"Tiny standard deviation ({g_std:.5f} < 0.01)"
            elif nonzero_ratio < 0.01:
                is_near_constant = True
                near_constant_reason = f"Extremely rare trigger (non-zero ratio {nonzero_ratio*100:.2f}% < 1.0%)"
            elif val_range > 0 and (g_std / val_range) < 0.01:
                is_near_constant = True
                near_constant_reason = f"Relative std negligible (std/range = {g_std/val_range:.5f})"

            if is_near_constant:
                near_constants.append({
                    "index": j,
                    "name": name,
                    "family": family,
                    "global_mean": g_mean,
                    "global_std": g_std,
                    "min": g_min,
                    "max": g_max,
                    "nonzero_ratio": nonzero_ratio,
                    "mean_within_std": mean_within_std,
                    "reason": near_constant_reason,
                })

        column_stats.append({
            "index": j,
            "name": name,
            "family": family,
            "global_mean": g_mean,
            "global_std": g_std,
            "min": g_min,
            "max": g_max,
            "nonzero_ratio": nonzero_ratio,
            "mean_within_std": mean_within_std,
            "max_within_std": max_within_std,
            "is_question_constant": is_q_constant,
            "is_near_constant": is_near_constant,
        })

    # 5. Pearson Correlation Matrix & Collinear Clustering (|r| >= 0.90)
    # Ensure zero std columns don't divide by zero
    valid_cols = [j for j in range(total_dim) if column_stats[j]["global_std"] > 1e-9]
    corr_matrix = np.corrcoef(x_flat[:, valid_cols], rowvar=False)

    collinear_pairs = []
    adj_graph = defaultdict(set)

    for idx1, c1 in enumerate(valid_cols):
        for idx2 in range(idx1 + 1, len(valid_cols)):
            c2 = valid_cols[idx2]
            r = corr_matrix[idx1, idx2]
            if abs(r) >= 0.90:
                collinear_pairs.append({
                    "col1_idx": c1,
                    "col1_name": feature_names[c1],
                    "col1_family": feature_families[c1],
                    "col2_idx": c2,
                    "col2_name": feature_names[c2],
                    "col2_family": feature_families[c2],
                    "r": float(r),
                })
                adj_graph[c1].add(c2)
                adj_graph[c2].add(c1)

    # Find connected components of collinear clusters
    visited = set()
    collinear_clusters = []

    for col in sorted(adj_graph.keys()):
        if col not in visited:
            component = []
            queue = [col]
            visited.add(col)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj_graph[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            component.sort()
            collinear_clusters.append(component)

    # 6. Recommendation for representative columns
    cluster_recommendations = []
    for cluster in collinear_clusters:
        c_names = [feature_names[c] for c in cluster]
        c_families = [feature_families[c] for c in cluster]
        
        # Designate best representative based on interpretability and variance
        # Default priority: normalized ratio > rank > raw count, or diff_mean > rank
        representative = None
        rep_reason = ""
        
        # Check known clusters:
        # Cluster A: cluster_count, cluster_ratio
        if "cluster_count" in c_names and "cluster_ratio" in c_names:
            representative = "cluster_ratio"
            rep_reason = "Normalized ratio handles variable valid answer counts across questions better than raw count"
        # Cluster B: char_count, word_count, line_count
        elif "char_count" in c_names and "word_count" in c_names:
            representative = "char_count"
            rep_reason = "Direct character length offers finest-grained measurement without tokenization noise"
        # Cluster C: char_count_diff_mean, word_count_diff_mean
        elif "char_count_diff_mean" in c_names and "word_count_diff_mean" in c_names:
            representative = "char_count_diff_mean"
            rep_reason = "Continuous zero-mean length deviation within question group"
        # Cluster D: char_count_rank, word_count_rank
        elif "char_count_rank" in c_names and "word_count_rank" in c_names:
            representative = "char_count_rank"
            rep_reason = "Discrete length rank in group"
        # Cluster E: num_equations, num_correct_equations
        elif "num_equations" in c_names and "num_correct_equations" in c_names:
            representative = "correct_equation_ratio" if "correct_equation_ratio" in c_names else "num_correct_equations"
            rep_reason = "Reflects successful derivation steps rather than noisy raw extraction count"
        # Cluster F: sim_other_mean, sim_centroid
        elif "sim_other_mean" in c_names or "sim_centroid" in c_names:
            representative = "sim_other_mean"
            rep_reason = "Direct pairwise cosine similarity to the other 3 candidate representations"
        # Cluster G: centroid_sim_diff_mean, dist_centroid_diff_mean
        elif "centroid_sim_diff_mean" in c_names and "dist_centroid_diff_mean" in c_names:
            representative = "centroid_sim_diff_mean"
            rep_reason = "Cosine similarity deviation has direct linear correlation with angle"
        else:
            # Fallback: choose the column with highest mean within-problem std
            stds = [column_stats[c]["mean_within_std"] for c in cluster]
            best_idx = int(np.argmax(stds))
            representative = c_names[best_idx]
            rep_reason = f"Highest within-problem discrimination (mean within std = {stds[best_idx]:.4f})"

        cluster_recommendations.append({
            "cluster_indices": cluster,
            "cluster_features": c_names,
            "cluster_families": c_families,
            "representative": representative,
            "reason": rep_reason,
        })

    # Output results
    results = {
        "dataset": "train_hard",
        "num_questions": n_q,
        "num_candidate_rows": n_rows,
        "total_feature_dimensions": total_dim,
        "question_level_constants": question_level_constants,
        "near_constants": near_constants,
        "collinear_pairs": collinear_pairs,
        "collinear_clusters": cluster_recommendations,
        "column_stats": column_stats,
    }
    results["column_contract"] = derive_column_contract(results)

    # Print summary tables
    print("\n" + "=" * 70)
    print("1. QUESTION-LEVEL CONSTANTS (Within-Problem Std == 0 for all questions)")
    print("   [Cannot distinguish or rank the 4 candidates within a question!]")
    print("=" * 70)
    for qc in question_level_constants:
        print(f"  - [{qc['family']}] {qc['name']} (idx={qc['index']}): Global Mean={qc['global_mean']:.4f}, Global Std={qc['global_std']:.4f}")

    print("\n" + "=" * 70)
    print("2. NEAR-CONSTANTS (Std < 0.01 or Trigger Rate < 1%)")
    print("   [Must NOT become core weights; high risk of spurious correlation]")
    print("=" * 70)
    for nc in near_constants:
        print(f"  - [{nc['family']}] {nc['name']} (idx={nc['index']}): Global Std={nc['global_std']:.4f}, Non-zero={nc['nonzero_ratio']*100:.2f}%. Reason: {nc['reason']}")

    print("\n" + "=" * 70)
    print(f"3. COLLINEAR PAIRS (|r| >= 0.90, Total = {len(collinear_pairs)} pairs)")
    print("=" * 70)
    for pair in collinear_pairs:
        print(f"  - {pair['col1_name']} ({pair['col1_family']}) <--> {pair['col2_name']} ({pair['col2_family']}): r = {pair['r']:+.4f}")

    print("\n" + "=" * 70)
    print(f"4. COLLINEARITY CLUSTERS & DESIGNATED REPRESENTATIVES FOR EXP-B03 (Total = {len(cluster_recommendations)} clusters)")
    print("=" * 70)
    for i, cl in enumerate(cluster_recommendations, 1):
        features_str = ", ".join(cl["cluster_features"])
        print(f"  Cluster {i}: [{features_str}]")
        print(f"    --> Recommended Representative: `{cl['representative']}` ({cl['reason']})")

    # Save to disk
    runs_dir = DEV_SRC_DIR.parent / 'runs' / 'audit_logs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_file = runs_dir / 'exp_b02_screening_train_hard.json'
    contract_file = runs_dir / 'private_configuration_omitted.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(contract_file, 'w', encoding='utf-8') as f:
        json.dump(results["column_contract"], f, indent=2, ensure_ascii=False)
    print(f"\nFull detailed audit saved to: {out_file}")
    print(f"Downstream column contract saved to: {contract_file}")

    return results


if __name__ == '__main__':
    run_exp_b02_screening()

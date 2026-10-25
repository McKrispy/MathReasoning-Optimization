"""EXP-B06: Pointwise Verifier on Raw [1024] Embeddings (Responding to RQ3).

Evaluates the performance upper-bound of assessing candidate solutions strictly in isolation (Pointwise),
without any cross-candidate comparison, group normalization, or consensus awareness.

1. Uses only raw 1024-dimensional candidate embeddings from arrays['embeddings'].
2. Implements MOD-M02:
   - Pointwise Logistic Regression (Linear Pointwise Baseline) with L2 regularization tuning.
   - Pointwise 2-layer MLP (Non-linear Pointwise Baseline: 1024 -> 128 -> 1).
3. Evaluates greedy candidate selection on:
   - val_hard (355 questions)
   - val_all (798 questions)
   - train_hard (2,114 questions)
4. Addresses Research Question 3 (RQ3):
   - Quantifies the performance gap between isolated candidate scoring vs cross-candidate comparison.
   - Analyzes why Pointwise verifiers struggle on hard reasoning problems.
5. Archives all artifacts to dev/runs/audit_logs/, dev/runs/predictions/, and dev/runs/scores/.
"""
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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


class PointwiseMLP(nn.Module):
    """Two-layer MLP for pointwise candidate verification."""
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def run_exp_b06() -> Dict[str, Any]:
    print("=" * 70)
    print("Executing EXP-B06: Pointwise Verifier on Raw [1024] Embeddings (RQ3)")
    print("=" * 70)

    # 1. Load data
    bank_train, arrays_train = load_dataset('train')
    gold_train = load_gold_references('train')
    train_hard_mask = get_hard_subset_mask(bank_train)

    bank_train_hard = [b for i, b in enumerate(bank_train) if train_hard_mask[i]]
    arrays_train_hard = {'embeddings': arrays_train['embeddings'][train_hard_mask]}
    labels_train_hard = build_label_matrix(bank_train_hard, [g for i, g in enumerate(gold_train) if train_hard_mask[i]])
    labels_train_all = build_label_matrix(bank_train, gold_train)

    bank_val, arrays_val = load_dataset('validation')
    gold_val = load_gold_references('validation')
    val_hard_mask = get_hard_subset_mask(bank_val)

    bank_val_hard = [b for i, b in enumerate(bank_val) if val_hard_mask[i]]
    arrays_val_hard = {'embeddings': arrays_val['embeddings'][val_hard_mask]}
    labels_val_hard = build_label_matrix(bank_val_hard, [g for i, g in enumerate(gold_val) if val_hard_mask[i]])
    labels_val_all = build_label_matrix(bank_val, gold_val)

    n_train_hard = len(bank_train_hard)
    n_train_all = len(bank_train)
    n_val_hard = len(bank_val_hard)
    n_val_all = len(bank_val)
    print(f"Loaded train_hard: {n_train_hard} questions ({n_train_hard*4} candidates)")
    print(f"Loaded val_hard:   {n_val_hard} questions ({n_val_hard*4} candidates)")
    print(f"Loaded val_all:    {n_val_all} questions ({n_val_all*4} candidates)")

    # 2. Extract strictly raw [1024] embeddings
    # L2-normalize vectors individually (pointwise operation)
    def normalize_l2(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (x / norms).astype(np.float32)

    emb_tr_hard = normalize_l2(arrays_train_hard['embeddings'])  # [2114, 4, 1024]
    emb_tr_all = normalize_l2(arrays_train['embeddings'])        # [4755, 4, 1024]
    emb_vh = normalize_l2(arrays_val_hard['embeddings'])         # [355, 4, 1024]
    emb_va = normalize_l2(arrays_val['embeddings'])              # [798, 4, 1024]

    # Flatten for pointwise training
    x_tr_hard_flat = emb_tr_hard.reshape(n_train_hard * 4, 1024)
    y_tr_hard_flat = labels_train_hard.reshape(-1)

    x_tr_all_flat = emb_tr_all.reshape(n_train_all * 4, 1024)
    y_tr_all_flat = labels_train_all.reshape(-1)

    # 3. Model 1: Pointwise L2 Logistic Regression (tune C)
    print("\n--- Training Model 1: Pointwise L2 Logistic Regression ---")
    c_candidates = [0.01, 0.1, 1.0, 10.0]
    best_lr = None
    best_lr_c = 1.0
    best_lr_acc_vh = -1.0
    lr_sweep_results = []

    for c in c_candidates:
        clf = LogisticRegression(C=c, solver='lbfgs', max_iter=500, random_state=42)
        clf.fit(x_tr_hard_flat, y_tr_hard_flat)

        # Evaluate on val_hard
        scores_vh = clf.decision_function(emb_vh.reshape(n_val_hard * 4, 1024)).reshape(n_val_hard, 4)
        pred_vh = np.argmax(scores_vh, axis=1)
        acc_vh = float(np.mean([labels_val_hard[i, pred_vh[i]] for i in range(n_val_hard)]))

        lr_sweep_results.append({"C": c, "val_hard_acc": acc_vh})
        print(f"  C = {c:<6}: val_hard accuracy = {acc_vh*100:.4f}% ({int(acc_vh*n_val_hard)} / {n_val_hard})")

        if acc_vh > best_lr_acc_vh:
            best_lr_acc_vh = acc_vh
            best_lr = clf
            best_lr_c = c

    print(f"Selected Best Pointwise Logistic Regression: C = {best_lr_c} (val_hard = {best_lr_acc_vh*100:.4f}%)")

    # Evaluate best LR on val_hard, val_all and train_hard
    scores_vh_lr = best_lr.decision_function(emb_vh.reshape(n_val_hard * 4, 1024)).reshape(n_val_hard, 4)
    pred_vh_lr = np.argmax(scores_vh_lr, axis=1)

    scores_va_lr = best_lr.decision_function(emb_va.reshape(n_val_all * 4, 1024)).reshape(n_val_all, 4)
    pred_va_lr = np.argmax(scores_va_lr, axis=1)
    acc_va_lr = float(np.mean([labels_val_all[i, pred_va_lr[i]] for i in range(n_val_all)]))

    scores_tr_lr = best_lr.decision_function(x_tr_hard_flat).reshape(n_train_hard, 4)
    pred_tr_lr = np.argmax(scores_tr_lr, axis=1)
    acc_tr_lr = float(np.mean([labels_train_hard[i, pred_tr_lr[i]] for i in range(n_train_hard)]))

    # 4. Model 2: Pointwise 2-Layer MLP (1024 -> 128 -> 1)
    print("\n--- Training Model 2: Pointwise 2-Layer MLP ---")
    torch.manual_seed(42)
    mlp = PointwiseMLP(input_dim=1024, hidden_dim=128, dropout=0.2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp.to(device)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    tensor_x_tr = torch.tensor(x_tr_hard_flat, dtype=torch.float32)
    tensor_y_tr = torch.tensor(y_tr_hard_flat, dtype=torch.float32)
    dataset_tr = torch.utils.data.TensorDataset(tensor_x_tr, tensor_y_tr)
    loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=256, shuffle=True)

    mlp.train()
    epochs = 15
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader_tr:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = mlp(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

    mlp.eval()
    with torch.no_grad():
        tensor_x_vh = torch.tensor(emb_vh.reshape(n_val_hard * 4, 1024), dtype=torch.float32).to(device)
        scores_vh_mlp = mlp(tensor_x_vh).cpu().numpy().reshape(n_val_hard, 4)
        pred_vh_mlp = np.argmax(scores_vh_mlp, axis=1)
        acc_vh_mlp = float(np.mean([labels_val_hard[i, pred_vh_mlp[i]] for i in range(n_val_hard)]))

        tensor_x_va = torch.tensor(emb_va.reshape(n_val_all * 4, 1024), dtype=torch.float32).to(device)
        scores_va_mlp = mlp(tensor_x_va).cpu().numpy().reshape(n_val_all, 4)
        pred_va_mlp = np.argmax(scores_va_mlp, axis=1)
        acc_va_mlp = float(np.mean([labels_val_all[i, pred_va_mlp[i]] for i in range(n_val_all)]))

        scores_tr_mlp = mlp(torch.tensor(x_tr_hard_flat, dtype=torch.float32).to(device)).cpu().numpy().reshape(n_train_hard, 4)
        pred_tr_mlp = np.argmax(scores_tr_mlp, axis=1)
        acc_tr_mlp = float(np.mean([labels_train_hard[i, pred_tr_mlp[i]] for i in range(n_train_hard)]))

    print(f"Pointwise MLP Results:")
    print(f"  val_hard accuracy: {acc_vh_mlp*100:.4f}% ({int(acc_vh_mlp*n_val_hard)} / {n_val_hard})")
    print(f"  val_all accuracy:  {acc_va_mlp*100:.4f}% ({int(acc_va_mlp*n_val_all)} / {n_val_all})")
    print(f"  train_hard acc:    {acc_tr_mlp*100:.4f}% ({int(acc_tr_mlp*n_train_hard)} / {n_train_hard})")

    # Select best pointwise model overall (between LR and MLP)
    if acc_vh_mlp >= best_lr_acc_vh:
        best_model_name = "Pointwise 2-Layer MLP (1024 -> 128 -> 1)"
        final_acc_vh = acc_vh_mlp
        final_acc_va = acc_va_mlp
        final_acc_tr = acc_tr_mlp
        final_pred_va = pred_va_mlp
        final_pred_vh = pred_vh_mlp
        final_scores_vh = scores_vh_mlp
        # Measure MLP model size
        import io
        buf = io.BytesIO()
        torch.save(mlp.state_dict(), buf)
        model_size_bytes = len(buf.getvalue())
    else:
        best_model_name = f"Pointwise Logistic Regression (C={best_lr_c})"
        final_acc_vh = best_lr_acc_vh
        final_acc_va = acc_va_lr
        final_acc_tr = acc_tr_lr
        final_pred_va = pred_va_lr
        final_pred_vh = pred_vh_lr
        final_scores_vh = scores_vh_lr
        import pickle
        model_size_bytes = len(pickle.dumps(best_lr))

    model_size_mib = model_size_bytes / (1024 * 1024)

    # Measure CPU inference latency on val_all
    t0 = time.perf_counter()
    if "MLP" in best_model_name:
        with torch.no_grad():
            _ = mlp(torch.tensor(emb_va.reshape(n_val_all * 4, 1024), dtype=torch.float32).to('cpu')).numpy()
    else:
        _ = best_lr.decision_function(emb_va.reshape(n_val_all * 4, 1024))
    latency_va = time.perf_counter() - t0

    # 5. Summary and Comparison with Baselines
    print("\n" + "=" * 70)
    print("EXP-B06 POINTWISE PERFORMANCE BENCHMARK (Responding to RQ3)")
    print("=" * 70)
    print(f"Selected Best Pointwise Model: {best_model_name}")
    print(f"  val_hard:   {final_acc_vh*100:.4f}% ({int(final_acc_vh*n_val_hard)} / {n_val_hard})")
    print(f"  val_all:    {final_acc_va*100:.4f}% ({int(final_acc_va*n_val_all)} / {n_val_all})")
    print(f"  train_hard: {final_acc_tr*100:.4f}% ({int(final_acc_tr*n_train_hard)} / {n_train_hard})")
    print(f"  Latency (val_all): {latency_va:.4f}s")
    print(f"  Asset Size:        {model_size_bytes} bytes ({model_size_mib:.4f} MiB)")

    print("\nBenchmark Comparisons on val_hard (355 questions):")
    print(f"  Random Baseline:          25.0000%")
    print(f"  Surface Heuristics (B05): 60.0000%")
    print(f"  Pointwise Embeddings(B06):{final_acc_vh*100:.4f}%")
    print(f"  Majority Voting (B01):    69.0141%")
    print(f"  Full Listwise Verifier:   73.2394%")

    gap_to_mv = final_acc_vh - 0.690141
    gap_to_listwise = final_acc_vh - 0.732394
    print(f"\nGap Analysis for RQ3:")
    print(f"  Pointwise vs Majority Voting: {gap_to_mv*100:+.2f}%")
    print(f"  Pointwise vs Listwise Model:  {gap_to_listwise*100:+.2f}%")

    # 6. Save audit artifacts
    runs_dir = DEV_SRC_DIR.parent / 'runs'
    audit_dir = runs_dir / 'audit_logs'
    pred_dir = runs_dir / 'predictions'
    score_dir = runs_dir / 'scores'

    audit_results = {
        "dataset": "val_hard (355 questions) / val_all (798 questions)",
        "features": "Raw [1024] Candidate Embeddings (Pointwise, zero cross-candidate features)",
        "best_model": best_model_name,
        "asset_size_bytes": model_size_bytes,
        "asset_size_mib": model_size_mib,
        "latency_seconds_val_all": latency_va,
        "lr_sweep": lr_sweep_results,
        "mlp_results": {
            "val_hard_acc": acc_vh_mlp,
            "val_all_acc": acc_va_mlp,
            "train_hard_acc": acc_tr_mlp,
        },
        "accuracy": {
            "val_hard": {
                "acc": final_acc_vh,
                "correct": int(round(final_acc_vh * n_val_hard)),
                "total": n_val_hard,
            },
            "val_all": {
                "acc": final_acc_va,
                "correct": int(round(final_acc_va * n_val_all)),
                "total": n_val_all,
            },
            "train_hard": {
                "acc": final_acc_tr,
                "correct": int(round(final_acc_tr * n_train_hard)),
                "total": n_train_hard,
            },
        },
        "rq3_comparisons": {
            "random_baseline": 0.25,
            "surface_b05": 0.6000,
            "pointwise_b06": final_acc_vh,
            "majority_voting_b01": 0.690141,
            "full_listwise_b03": 0.732394,
            "pointwise_gap_to_majority_voting": gap_to_mv,
            "pointwise_gap_to_listwise": gap_to_listwise,
        },
    }

    audit_out = audit_dir / 'exp_b06_pointwise_audit.json'
    with open(audit_out, 'w', encoding='utf-8') as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)
    print(f"\nAudit saved to: {audit_out}")

    pred_out = pred_dir / 'EXP-B06_pointwise_predictions.json'
    pred_data = {
        "split": "validation",
        "predictions": [int(p) for p in final_pred_va],
    }
    with open(pred_out, 'w', encoding='utf-8') as f:
        json.dump(pred_data, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to: {pred_out}")

    score_out = score_dir / 'EXP-B06_pointwise_score.json'
    score_data = {
        "split": "validation",
        "total": n_val_all,
        "correct": int(round(final_acc_va * n_val_all)),
        "accuracy": final_acc_va,
        "val_hard_total": n_val_hard,
        "val_hard_correct": int(round(final_acc_vh * n_val_hard)),
        "val_hard_accuracy": final_acc_vh,
    }
    with open(score_out, 'w', encoding='utf-8') as f:
        json.dump(score_data, f, indent=2, ensure_ascii=False)
    print(f"Score saved to: {score_out}")

    return audit_results


if __name__ == '__main__':
    run_exp_b06()

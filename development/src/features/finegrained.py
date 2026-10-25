"""FEAT-G05: Fine-Grained Symbolic Chain and Cluster Evidence Feature Extractor.

Extracts fine-grained symbolic chain continuity, final equation agreement,
Markdown heading conflict traps, and intra-problem answer cluster collective verification evidence.
Directly addresses Research Question 1 (RQ1) and Research Question 2 (RQ2).
"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure official code is importable
OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[3] / "DASE7506-MP2" / "code"
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.append(str(OFFICIAL_CODE_DIR))

import common  # type: ignore
from .base import BaseFeatureExtractor
from .symbolic import (
    extract_numbers_from_text,
    normalize_trace_text,
    safe_eval_expr,
)
from postprocessing.masking import detect_parser_heading_conflict


def extract_equation_records(text: str) -> List[Dict[str, Any]]:
    """Extract and parse all arithmetic equations in candidate trace with operand tracking.
    
    Returns:
        List of dicts containing:
          - lhs_clean: cleaned LHS expression string
          - lhs_val: evaluated LHS numerical value
          - rhs_val: target RHS numerical value
          - is_correct: True if equation evaluates within numerical tolerance
          - operands: list of numerical operands extracted from the LHS
    """
    norm_text = normalize_trace_text(text)
    eq_pattern = re.compile(
        r"([0-9\.\s\+\-\*\/\(\)]+[\+\-\*\/][0-9\.\s\+\-\*\/\(\)]+)\s*(?:=|\\approx|≈)\s*([-+]?\d+(?:\.\d+)?)"
    )
    records: List[Dict[str, Any]] = []

    for match in eq_pattern.finditer(norm_text):
        lhs_raw, rhs_raw = match.group(1).strip(), match.group(2).strip()
        lhs_clean = re.sub(r"^[^\d\(\+\-]+|[^\d\)\s]+$", "", lhs_raw).strip()
        lhs_val = safe_eval_expr(lhs_clean)
        if lhs_val is None:
            continue
        try:
            rhs_val = float(rhs_raw)
        except ValueError:
            continue

        abs_err = abs(lhs_val - rhs_val)
        rel_err = abs_err / max(abs(rhs_val), 1e-4)
        is_corr = bool(abs_err < 0.05 or rel_err < 0.05)

        op_matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", lhs_clean)
        operands: List[float] = []
        for op_m in op_matches:
            try:
                operands.append(float(op_m.replace(",", "")))
            except ValueError:
                pass

        records.append({
            "lhs": lhs_clean,
            "lhs_val": lhs_val,
            "rhs_val": rhs_val,
            "is_correct": is_corr,
            "operands": operands,
        })
    return records


class FineGrainedVerificationFeatureExtractor(BaseFeatureExtractor):
    """FEAT-G05: Fine-Grained Symbolic Chain & Answer Cluster Evidence Feature Extractor.

    Features extracted per candidate (6 dimensions):
      [Fine-Grained Symbolic & Format Verification]
       0: final_eq_matches_answer (1.0 if RHS of last verified equation matches parsed answer, else 0.0)
       1: equation_chain_continuity (fraction of operands in eq k>=2 originating from question stem or prior RHS)
       2: has_parser_heading_conflict (1.0 if candidate contains Markdown heading conflict trap, else 0.0)

      [Intra-Problem Answer Cluster Collective Evidence]
       3: cluster_avg_correct_eq_ratio (mean correct equation ratio across candidates in same answer cluster)
       4: cluster_max_condition_coverage (max question condition coverage across candidates in same answer cluster)
       5: cluster_min_wrong_eq (min wrong equation count across candidates in same answer cluster)
    """

    FEATURE_NAMES = [
        "final_eq_matches_answer",
        "equation_chain_continuity",
        "has_parser_heading_conflict",
        "cluster_avg_correct_eq_ratio",
        "cluster_max_condition_coverage",
        "cluster_min_wrong_eq",
    ]

    def __init__(self):
        super().__init__(name="FEAT-G05-finegrained")

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def transform(self, bank: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(bank)
        feats = np.zeros((n, 4, self.feature_dim), dtype=np.float32)

        for i, row in enumerate(bank):
            q_text = row.get("question", "")
            q_numbers = extract_numbers_from_text(q_text)
            q_num_floats = [float(f) for f in q_numbers]
            q_num_count = len(q_numbers)

            cands = row["candidates"]
            c_records: List[List[Dict[str, Any]]] = []
            c_parsed: List[Optional[str]] = []
            c_cov: List[float] = []
            c_corr_ratio: List[float] = []
            c_wrong_cnt: List[int] = []

            for c in range(4):
                text = cands[c].get("text", "")
                parsed = common.parse_answer(text)
                c_parsed.append(parsed)

                eqs = extract_equation_records(text)
                c_records.append(eqs)

                n_eq = len(eqs)
                n_corr = sum(1 for e in eqs if e["is_correct"])
                n_wrong = n_eq - n_corr
                corr_ratio = float(n_corr / n_eq) if n_eq > 0 else 1.0

                c_numbers = extract_numbers_from_text(text)
                if q_num_count > 0:
                    cov = float(len(q_numbers & c_numbers) / q_num_count)
                else:
                    cov = 1.0

                c_cov.append(cov)
                c_corr_ratio.append(corr_ratio)
                c_wrong_cnt.append(n_wrong)

            # Build cluster map by parsed answer
            cluster_indices = defaultdict(list)
            for c in range(4):
                p = c_parsed[c]
                if p is not None:
                    cluster_indices[p].append(c)

            for c in range(4):
                text = cands[c].get("text", "")
                eqs = c_records[c]
                parsed = c_parsed[c]

                # 0: final_eq_matches_answer
                if eqs and parsed is not None:
                    try:
                        p_val = float(Fraction(parsed))
                        last_rhs = eqs[-1]["rhs_val"]
                        if (
                            abs(last_rhs - p_val) < 1e-3
                            or abs(last_rhs - p_val) / max(abs(p_val), 1e-4) < 1e-3
                        ):
                            feats[i, c, 0] = 1.0
                        else:
                            feats[i, c, 0] = 0.0
                    except (ValueError, ZeroDivisionError):
                        feats[i, c, 0] = 0.0
                else:
                    feats[i, c, 0] = 0.0

                # 1: equation_chain_continuity
                if len(eqs) <= 1:
                    feats[i, c, 1] = 1.0
                else:
                    known = set(q_num_floats)
                    known.add(eqs[0]["rhs_val"])
                    total_ops = 0
                    matched_ops = 0
                    for eq in eqs[1:]:
                        for op in eq["operands"]:
                            total_ops += 1
                            if any(
                                abs(op - k) < 1e-2
                                or abs(op - k) / max(abs(k), 1e-4) < 1e-2
                                for k in known
                            ):
                                matched_ops += 1
                        known.add(eq["rhs_val"])
                    feats[i, c, 1] = float(matched_ops / total_ops) if total_ops > 0 else 1.0

                # 2: has_parser_heading_conflict
                feats[i, c, 2] = 1.0 if detect_parser_heading_conflict(text) else 0.0

                # 3, 4, 5: cluster collective verification features
                if parsed is not None and parsed in cluster_indices:
                    members = cluster_indices[parsed]
                    feats[i, c, 3] = float(np.mean([c_corr_ratio[m] for m in members]))
                    feats[i, c, 4] = float(np.max([c_cov[m] for m in members]))
                    feats[i, c, 5] = float(np.min([c_wrong_cnt[m] for m in members]))
                else:
                    feats[i, c, 3] = c_corr_ratio[c]
                    feats[i, c, 4] = c_cov[c]
                    feats[i, c, 5] = float(c_wrong_cnt[c])

        return feats

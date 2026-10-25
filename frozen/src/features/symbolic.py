"""FEAT-G03: Reasoning arithmetic verification and question condition coverage extractor.

Parses explicit arithmetic equations in candidate traces, verifies numerical validity via safe evaluation,
and checks coverage of numbers mentioned in the question prompt. Directly addresses Research Question 1 (RQ1).
"""
import ast
from fractions import Fraction
import operator
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

# Ensure official code is importable
OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[3] / 'code'
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.append(str(OFFICIAL_CODE_DIR))

from .base import BaseFeatureExtractor

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval_ast(node: ast.AST) -> float:
    """Safely evaluate arithmetic expression AST without arbitrary code execution."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Non-numeric constant")
    elif isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        operand = safe_eval_ast(node.operand)
        return float(SAFE_OPERATORS[type(node.op)](operand))
    elif isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        left = safe_eval_ast(node.left)
        right = safe_eval_ast(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv)) and abs(right) < 1e-12:
            raise ZeroDivisionError("Division by zero in trace")
        return float(SAFE_OPERATORS[type(node.op)](left, right))
    raise ValueError("Unsupported AST node in equation")


def safe_eval_expr(expr_str: str) -> Optional[float]:
    """Parse and safely evaluate a simple arithmetic string (e.g. '10 + 5 * 2')."""
    clean = expr_str.strip()
    try:
        parsed = ast.parse(clean, mode='eval')
        return safe_eval_ast(parsed.body)
    except Exception:
        return None


def extract_numbers_from_text(text: str) -> Set[Fraction]:
    """Extract all numerical values appearing in text as Fractions."""
    matches = re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', text)
    numbers = set()
    for m in matches:
        clean = m.replace(',', '')
        try:
            numbers.add(Fraction(clean))
        except (ValueError, ZeroDivisionError):
            pass
    return numbers


def normalize_trace_text(text: str) -> str:
    """Normalize mathematical symbols in text for equation extraction."""
    # Convert LaTeX fractions \frac{A}{B} -> (A)/(B)
    t = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', text)
    # Replace multiplication variants
    t = re.sub(r'\\times|\\cdot|×', '*', t)
    # Replace division variants
    t = re.sub(r'\\div|÷', '/', t)
    # Remove currency symbols and formatting commas in numbers
    t = re.sub(r'\$', '', t)
    t = re.sub(r'(?<=\d),(?=\d)', '', t)
    return t


def extract_and_verify_equations(text: str) -> Tuple[int, int, int]:
    """Extract and verify equations in candidate trace.
    
    Returns:
        (num_equations, num_correct, num_wrong)
    """
    norm_text = normalize_trace_text(text)
    # Match pattern: LHS = RHS or LHS \approx RHS
    # LHS must have at least one operator (+, -, *, /) and numbers
    # RHS is a single number
    eq_pattern = re.compile(
        r'([0-9\.\s\+\-\*\/\(\)]+[\+\-\*\/][0-9\.\s\+\-\*\/\(\)]+)\s*(?:=|\\approx|≈)\s*([-+]?\d+(?:\.\d+)?)'
    )

    num_equations = 0
    num_correct = 0
    num_wrong = 0

    for match in eq_pattern.finditer(norm_text):
        lhs_raw, rhs_raw = match.group(1).strip(), match.group(2).strip()

        # Clean up leading/trailing punctuation in LHS
        lhs_clean = re.sub(r'^[^\d\(\+\-]+|[^\d\)\s]+$', '', lhs_raw).strip()
        lhs_val = safe_eval_expr(lhs_clean)

        if lhs_val is None:
            continue

        try:
            rhs_val = float(rhs_raw)
        except ValueError:
            continue

        num_equations += 1
        # Allow small floating point tolerance or rounding (5% relative error or 0.05 absolute)
        abs_err = abs(lhs_val - rhs_val)
        rel_err = abs_err / max(abs(rhs_val), 1e-4)

        if abs_err < 0.05 or rel_err < 0.05:
            num_correct += 1
        else:
            num_wrong += 1

    return num_equations, num_correct, num_wrong


class SymbolicVerificationFeatureExtractor(BaseFeatureExtractor):
    """FEAT-G03: Reasoning & Symbolic Verification Feature Extractor.
    
    Features extracted per candidate (14 dimensions):
      [Arithmetic Equation Verification]
       0: num_equations (total valid arithmetic equations found)
       1: num_correct_equations (equations evaluating correctly)
       2: num_wrong_equations (equations with calculation errors)
       3: correct_equation_ratio (correct / total equations, 1.0 if none found)
       4: has_arithmetic_error (1.0 if any arithmetic error found)
      
      [Question Condition Alignment]
       5: question_number_count (count of distinct numbers in question prompt)
       6: candidate_number_count (count of distinct numbers in candidate trace)
       7: question_condition_coverage (fraction of question numbers present in trace)
       8: missing_condition_count (question numbers absent from trace)
       9: has_missing_condition (1.0 if any question number is absent)
      
      [Group-Relative Comparative Features across the 4 candidates]
      10: wrong_eq_diff_mean (candidate wrong equations minus group mean)
      11: wrong_eq_rank (candidate rank 0..3 by wrong equations; lower is better)
      12: condition_coverage_diff_mean (candidate coverage minus group mean)
      13: condition_coverage_rank (candidate rank 0..3 by coverage; higher is better)
    """

    FEATURE_NAMES = [
        "num_equations",
        "num_correct_equations",
        "num_wrong_equations",
        "correct_equation_ratio",
        "has_arithmetic_error",
        "question_number_count",
        "candidate_number_count",
        "question_condition_coverage",
        "missing_condition_count",
        "has_missing_condition",
        "wrong_eq_diff_mean",
        "wrong_eq_rank",
        "condition_coverage_diff_mean",
        "condition_coverage_rank",
    ]

    def __init__(self):
        super().__init__(name="FEAT-G03-symbolic")

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def transform(self, bank: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(bank)
        feats = np.zeros((n, 4, self.feature_dim), dtype=np.float32)

        for i, row in enumerate(bank):
            q_numbers = extract_numbers_from_text(row.get('question', ''))
            q_num_count = len(q_numbers)

            group_wrong_eq = np.zeros(4, dtype=np.float32)
            group_coverage = np.zeros(4, dtype=np.float32)

            for c in range(4):
                text = row['candidates'][c].get('text', '')

                # 1. Arithmetic verification
                n_eq, n_corr, n_wrong = extract_and_verify_equations(text)
                corr_ratio = float(n_corr / n_eq) if n_eq > 0 else 1.0
                has_err = 1.0 if n_wrong > 0 else 0.0

                # 2. Condition coverage
                c_numbers = extract_numbers_from_text(text)
                c_num_count = len(c_numbers)

                if q_num_count > 0:
                    covered = len(q_numbers & c_numbers)
                    coverage = float(covered / q_num_count)
                    missing = q_num_count - covered
                else:
                    coverage = 1.0
                    missing = 0

                has_missing = 1.0 if missing > 0 else 0.0

                feats[i, c, 0] = float(n_eq)
                feats[i, c, 1] = float(n_corr)
                feats[i, c, 2] = float(n_wrong)
                feats[i, c, 3] = corr_ratio
                feats[i, c, 4] = has_err
                feats[i, c, 5] = float(q_num_count)
                feats[i, c, 6] = float(c_num_count)
                feats[i, c, 7] = coverage
                feats[i, c, 8] = float(missing)
                feats[i, c, 9] = has_missing

                group_wrong_eq[c] = float(n_wrong)
                group_coverage[c] = coverage

            # 3. Group relative comparisons
            feats[i, :, 10] = group_wrong_eq - group_wrong_eq.mean()
            # Wrong equations: lower is better (0 = least errors, 3 = most errors)
            feats[i, :, 11] = np.argsort(np.argsort(group_wrong_eq)).astype(np.float32)

            feats[i, :, 12] = group_coverage - group_coverage.mean()
            # Coverage: higher is better (3 = highest coverage)
            feats[i, :, 13] = np.argsort(np.argsort(group_coverage)).astype(np.float32)

        return feats

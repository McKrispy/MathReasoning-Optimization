"""Test for EXP-B02 feature screening reproducibility."""
from pathlib import Path
import sys
import json
import numpy as np

# Ensure dev/src is in sys.path
DEV_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b02_screening import run_exp_b02_screening


def test_exp_b02_screening_integrity():
    results = run_exp_b02_screening()
    
    assert results['dataset'] == 'train_hard'
    assert results['num_questions'] == 2114
    assert results['num_candidate_rows'] == 8456
    assert results['total_feature_dimensions'] == 56

    # Verify 3 question-level constants identified
    q_const_names = {qc['name'] for qc in results['question_level_constants']}
    assert 'num_distinct_answers' in q_const_names
    assert 'question_number_count' in q_const_names
    assert 'has_diff_answer_other' in q_const_names

    # Verify collinear clusters found
    assert len(results['collinear_clusters']) == 8
    
    # Verify audit log exists
    audit_file = DEV_SRC.parent / 'runs' / 'audit_logs' / 'exp_b02_screening_train_hard.json'
    assert audit_file.exists()

"""TEST-T05: Offline determinism and reproducibility test."""
import filecmp
import json
import os
from pathlib import Path
import subprocess
import sys

DEV_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEV_ROOT.parent
OFFICIAL_CODE_DIR = PROJECT_ROOT / 'DASE7506-MP2' / 'code'


def test_prediction_reproducibility(split: str = 'validation'):
    out_1 = DEV_ROOT / 'runs' / 'predictions' / f'reproduce_1_{split}.json'
    out_2 = DEV_ROOT / 'runs' / 'predictions' / f'reproduce_2_{split}.json'

    cmd = [
        sys.executable,
        str(OFFICIAL_CODE_DIR / 'predict.py'),
        '--method', str(DEV_ROOT / 'src' / 'submission.py'),
        '--split', split,
    ]

    # Run 1
    subprocess.run(cmd + ['--output', str(out_1)], check=True, cwd=str(PROJECT_ROOT))
    # Run 2
    subprocess.run(cmd + ['--output', str(out_2)], check=True, cwd=str(PROJECT_ROOT))

    data1 = json.loads(out_1.read_text())
    data2 = json.loads(out_2.read_text())

    assert data1['selected'] == data2['selected'], "Non-deterministic predictions detected!"
    assert data1['ids'] == data2['ids']


if __name__ == "__main__":
    test_prediction_reproducibility('validation')
    print("TEST-T05 PASSED: Determinism verified.")

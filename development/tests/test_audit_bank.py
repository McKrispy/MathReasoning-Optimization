"""TEST-T04: Audit bank blind test and anti-hardcoding compliance test."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np

DEV_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEV_ROOT.parent
OFFICIAL_CODE_DIR = PROJECT_ROOT / 'DASE7506-MP2' / 'code'

if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_DIR))

import common  # type: ignore


def test_mock_audit_bank():
    """Create a mock audit bank with arbitrary size and verify predict runs without gold reference."""
    # Sample 15 items from test split
    bank, arrays = common.load_inputs('test')
    mock_n = 15
    mock_bank = bank[:mock_n]
    # Scramble IDs to test anti-lookup
    for i, row in enumerate(mock_bank):
        row = dict(row)
        row['id'] = f'audit_mock_{i:04d}'
        mock_bank[i] = row
    mock_embeddings = arrays['embeddings'][:mock_n].reshape(4 * mock_n, 1024)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cand_dir = tmp_path / 'candidates'
        cand_dir.mkdir(parents=True, exist_ok=True)

        # Write candidates jsonl
        cand_file = cand_dir / 'gsm8k_audit.jsonl'
        with open(cand_file, 'w', encoding='utf-8') as f:
            for row in mock_bank:
                f.write(json.dumps(row) + '\n')

        # Write embeddings npz
        npz_file = cand_dir / 'gsm8k_audit.npz'
        np.savez(npz_file, embeddings=mock_embeddings)

        # Create manifest
        manifest = {
            'protocol': 'dase7506-mp2-open-hard-verifier-v5',
            'splits': {'audit': mock_n},
            'sha256': {
                'candidates/gsm8k_audit.jsonl': common.hashlib.sha256(cand_file.read_bytes()).hexdigest(),
                'candidates/gsm8k_audit.npz': common.hashlib.sha256(npz_file.read_bytes()).hexdigest(),
            }
        }
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest, indent=2))

        pred_out = DEV_ROOT / 'runs' / 'predictions' / 'audit_mock_predictions.json'
        cmd = [
            sys.executable,
            str(OFFICIAL_CODE_DIR / 'predict.py'),
            '--method', str(DEV_ROOT / 'src' / 'submission.py'),
            '--split', 'audit',
            '--data-dir', str(tmp_path),
            '--output', str(pred_out),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Audit prediction failed: {res.stderr}"

        # Check prediction results
        res_data = json.loads(pred_out.read_text())
        assert len(res_data['selected']) == mock_n
        assert all(0 <= idx < 4 for idx in res_data['selected'])


if __name__ == "__main__":
    test_mock_audit_bank()
    print("TEST-T04 PASSED: Audit bank blind test completed successfully.")

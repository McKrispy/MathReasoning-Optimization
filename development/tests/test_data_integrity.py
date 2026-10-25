"""TEST-T01: Official data integrity and hash verification test."""
from pathlib import Path
import sys

OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[2] / 'DASE7506-MP2' / 'code'
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_DIR))

import common  # type: ignore


def test_official_data_hashes():
    """Verify that all official data files match their SHA256 digests in manifest.json."""
    manifest = common.check_data(data_dir=common.DATA)
    assert manifest['protocol'] == 'dase7506-mp2-open-hard-verifier-v5'
    assert manifest['splits']['train'] == 4755
    assert manifest['splits']['validation'] == 798
    assert manifest['splits']['test'] == 467


def test_split_disjointness():
    """Verify that question IDs across train, validation, and test are completely disjoint."""
    train_bank = common.load_bank('train')
    val_bank = common.load_bank('validation')
    test_bank = common.load_bank('test')

    train_ids = {r['id'] for r in train_bank}
    val_ids = {r['id'] for r in val_bank}
    test_ids = {r['id'] for r in test_bank}

    assert train_ids.isdisjoint(val_ids), "Train and validation splits overlap!"
    assert train_ids.isdisjoint(test_ids), "Train and test splits overlap!"
    assert val_ids.isdisjoint(test_ids), "Validation and test splits overlap!"


if __name__ == "__main__":
    test_official_data_hashes()
    test_split_disjointness()
    print("TEST-T01 PASSED: Official data integrity & split disjointness verified.")

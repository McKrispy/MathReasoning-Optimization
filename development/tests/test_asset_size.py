"""TEST-T02: Model asset size compliance test (strict <= 16 MiB limit)."""
from pathlib import Path
import json

MAX_BYTES = 16 * 1024 * 1024  # 16 MiB = 16,777,216 bytes
MODEL_BUNDLE_DIR = Path(__file__).resolve().parents[1] / 'runs' / 'model_bundle'


def get_bundle_size_bytes(directory: Path) -> int:
    """Recursively calculate uncompressed size of all files in directory."""
    if not directory.exists():
        return 0
    total_bytes = 0
    for path in directory.rglob('*'):
        if path.is_file() and path.name != '.gitkeep':
            total_bytes += path.stat().st_size
    return total_bytes


def test_asset_size_limit():
    total_bytes = get_bundle_size_bytes(MODEL_BUNDLE_DIR)
    size_mib = total_bytes / (1024 * 1024)
    print(f"Model bundle total uncompressed size: {total_bytes} bytes ({size_mib:.3f} MiB)")
    assert total_bytes <= MAX_BYTES, f"Model bundle exceeds 16 MiB: {size_mib:.2f} MiB > 16.0 MiB"


if __name__ == "__main__":
    test_asset_size_limit()
    print("TEST-T02 PASSED: Asset size is within 16 MiB limit.")

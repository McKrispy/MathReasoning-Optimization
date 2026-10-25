"""Install the B29 prediction guard when the sealed launcher enables it."""
import os
import sys
from pathlib import Path

if os.environ.get("B29_GUARD") == "1":
    src = Path(__file__).resolve().parents[1]
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    import b29_guard

    b29_guard.install_from_env()

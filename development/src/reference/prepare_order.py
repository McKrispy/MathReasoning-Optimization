"""EXP-R01 step 1 (env mp1): validation question order and val_hard flags.

Only candidate texts are read; reference answers are not opened. The output lets
the judge script (env mp1ref, no numpy) run val_hard questions first.
"""
import json
import sys
from pathlib import Path

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

from common_utils import get_hard_subset_mask, load_dataset  # noqa: E402

REF_RUNS = DEV_SRC_DIR.parent / "runs" / "ref"


def main() -> None:
    bank, _ = load_dataset("validation")
    hard = get_hard_subset_mask(bank)
    order = [{"id": row["id"], "hard": bool(h)} for row, h in zip(bank, hard)]
    REF_RUNS.mkdir(parents=True, exist_ok=True)
    out = REF_RUNS / "validation_order.json"
    out.write_text(json.dumps({"split": "validation", "questions": order}, indent=1), encoding="utf-8")
    print(f"{len(order)} questions, {int(hard.sum())} val_hard -> {out}")


if __name__ == "__main__":
    main()

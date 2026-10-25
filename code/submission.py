"""Public entrypoint for the frozen B17/B20 selector."""

from pathlib import Path
import sys

FROZEN_SOURCE = Path(__file__).resolve().parents[1] / "frozen" / "src"
if str(FROZEN_SOURCE) not in sys.path:
    sys.path.insert(0, str(FROZEN_SOURCE))

from inference import predict as _predict


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    return _predict(bank, arrays, checkpoint)

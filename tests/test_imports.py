import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_imports():
    import app.main  # noqa: F401
    import src.inference  # noqa: F401
    import src.train_classifier  # noqa: F401
    import src.train_detector  # noqa: F401

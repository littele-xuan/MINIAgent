from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_all_tests import main as run_all_main  # noqa: E402


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--real-only", "--require-real"]
    raise SystemExit(run_all_main())

"""Start the read-only paper dashboard on http://127.0.0.1:8765

    uv run python scripts/dashboard.py [--port 8765]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402

from fedge.dashboard.app import create_app  # noqa: E402

PORT = 8765


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    # Loopback only: the dashboard has no auth.
    uvicorn.run(create_app(ROOT), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

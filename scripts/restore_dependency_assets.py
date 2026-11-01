"""Restore only missing ignored assets from upstream/main without staging."""
import subprocess
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--source", default="c353731")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
for relative in (root / "docs/dependency_assets.txt").read_text().splitlines():
    target = root / relative
    if target.exists() or target.is_symlink():
        continue
    content = subprocess.check_output(["git", "show", args.source + ":" + relative], cwd=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(content)

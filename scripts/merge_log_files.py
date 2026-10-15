"""Merge the log files listed in a run's log_files.txt, in listed order."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def merge_log_files(result_dir: Path, output: Path | None = None) -> Path:
    manifest = result_dir / "log_files.txt"
    entries = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"No log files listed in {manifest}")

    log_paths = [
        path if path.is_absolute() else PROJECT_ROOT / path
        for path in map(Path, entries)
    ]
    for path in log_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Log file not found: {path}")

    output = output or result_dir / "merged.log"
    if output.resolve() in {path.resolve() for path in log_paths}:
        raise ValueError(f"Output would overwrite an input log: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as merged:
            temporary_path = Path(merged.name)
            for path in log_paths:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, merged)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Run directory containing log_files.txt")
    parser.add_argument("--output", type=Path, help="Output path (default: result_dir/merged.log)")
    args = parser.parse_args()
    print(merge_log_files(args.result_dir, args.output))

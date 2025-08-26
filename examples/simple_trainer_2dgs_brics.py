#!/usr/bin/env python3
"""
Batch launcher for 2DGS training over a BRiCS-style base directory.

Scans: <BASE_DIR>/<YYYY-MM-DD>/multisequence<number>/calib/stage2
If stage2 exists, launches:
  python simple_trainer_2dgs.py \
    --data-dir <...>/calib/stage2 \
    --data-factor 1 \
    --result-dir <...>/gsplat_2dgs \
    --disable-viewer

Examples:
  python -m examples.simple_trainer_2dgs_brics \
    --base_dir /mnt/brics-studio

  python -m examples.simple_trainer_2dgs_brics \
    --base_dir /mnt/brics-studio \
    --date 2025-03-31 --multiseq multisequence000001
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def is_date_folder(p: Path) -> bool:
    n = p.name
    return (
        p.is_dir()
        and len(n) == 10
        and n[4] == "-"
        and n[7] == "-"
        and n.replace("-", "").isdigit()
    )


def find_targets(base_dir: Path, date: Optional[str], multiseq: Optional[str]) -> List[Tuple[Path, Path, Path]]:
    """
    Returns a list of tuples: (stage2_dir, result_dir, multiseq_dir)
    where stage2_dir = <base>/<date>/<multiseq>/calib/stage2
          result_dir = <base>/<date>/<multiseq>/gsplat_2dgs
    Filtered by optional date and multiseq.
    """
    targets: List[Tuple[Path, Path, Path]] = []
    if not base_dir.exists():
        return targets
    date_dirs = sorted([d for d in base_dir.iterdir() if is_date_folder(d)])
    for d in date_dirs:
        if date and d.name != date:
            continue
        # multisequence folders
        for m in sorted(d.iterdir()):
            if not m.is_dir():
                continue
            if not m.name.startswith("multisequence"):
                continue
            if multiseq and m.name != multiseq:
                continue
            stage2 = m / "calib" / "stage2"
            if stage2.exists():
                result_dir = m / "gsplat_2dgs"
                targets.append((stage2, result_dir, m))
    return targets


def build_cmd(python_exe: str, trainer: Path, stage2: Path, result_dir: Path, data_factor: int, extra: Iterable[str]) -> List[str]:
    cmd = [
        python_exe,
        str(trainer),
        "--data-dir",
        str(stage2),
        "--data-factor",
        str(data_factor),
        "--result-dir",
        str(result_dir),
        "--disable-viewer",
    ]
    cmd.extend(extra)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch 2DGS trainer over BRiCS base directory")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing <YYYY-MM-DD>/multisequence*/calib/stage2")
    parser.add_argument("--date", type=str, default=None, help="Optional date folder to process (YYYY-MM-DD)")
    parser.add_argument("--multiseq", type=str, default=None, help="Optional multisequence folder to process (e.g., multisequence000001)")
    parser.add_argument("--data-factor", type=int, default=1, help="Value for --data-factor")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if result dir already contains ckpts")
    parser.add_argument("--extra", type=str, nargs=argparse.REMAINDER, help="Extra args to pass to simple_trainer_2dgs.py (prefix with --)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser()
    # Resolve trainer script path relative to this file
    script_dir = Path(__file__).resolve().parent
    trainer_script = script_dir / "simple_trainer_2dgs.py"
    if not trainer_script.exists():
        print(f"[error] Trainer script not found: {trainer_script}")
        return 1

    targets = find_targets(base_dir, args.date, args.multiseq)
    if not targets:
        print(f"[info] No targets found under: {base_dir}")
        return 0

    python_exe = sys.executable
    extra = args.extra or []

    ran = 0
    for stage2, result_dir, mdir in targets:
        # Optionally skip if result_dir already has checkpoints
        if args.skip_existing:
            ckpts = list((result_dir / "ckpts").glob("ckpt_*.pt"))
            if ckpts:
                print(f"[skip] {mdir} (existing {len(ckpts)} ckpt(s))")
                continue
        cmd = build_cmd(python_exe, trainer_script, stage2, result_dir, args.__dict__["data-factor"], extra)
        print("[run]", " ".join(cmd))
        if args.dry_run:
            continue
        # Ensure result dir exists
        try:
            result_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        proc = subprocess.run(cmd, cwd=str(script_dir))
        if proc.returncode != 0:
            print(f"[error] Training failed for: {mdir} (exit={proc.returncode})")
            return proc.returncode
        ran += 1

    print(f"[done] Launched {ran} training job(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

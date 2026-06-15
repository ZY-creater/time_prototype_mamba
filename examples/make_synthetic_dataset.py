from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from time_prototype_mamba.data.synthetic import create_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic longitudinal volume dataset.")
    parser.add_argument("--out", default="data/synthetic", help="Output directory relative to this repository.")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shape", nargs=3, type=int, default=(16, 24, 24), metavar=("D", "H", "W"))
    parser.add_argument("--max-timepoints", type=int, default=4)
    args = parser.parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    paths = create_synthetic_dataset(
        out_dir,
        num_samples=args.num_samples,
        shape_zyx=tuple(args.shape),
        max_timepoints=args.max_timepoints,
        seed=args.seed,
    )
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()


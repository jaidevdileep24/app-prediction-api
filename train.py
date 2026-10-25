"""Train the model: `python train.py` (synthetic demo data) or `python train.py --csv data.csv`."""
import argparse

from app.predictor import DEFAULT_MODEL_PATH
from app.training import train_and_save


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", help="anonymised in-app data with the feature columns and a 0/1 'purchased' label")
    ap.add_argument("--out", default=str(DEFAULT_MODEL_PATH), help="where to save the model bundle")
    args = ap.parse_args(argv)
    bundle = train_and_save(args.out, args.csv)
    print(f"Saved {args.out} | version {bundle['version']} | metrics {bundle['metrics']}")
    return bundle


if __name__ == "__main__":
    main()

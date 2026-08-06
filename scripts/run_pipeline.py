"""PumpPredictor pipeline CLI.

Usage:
    python scripts/run_pipeline.py universe     # screen the penny-stock universe
    python scripts/run_pipeline.py download     # fetch/cache OHLCV history
    python scripts/run_pipeline.py dataset      # build the labeled feature matrix
    python scripts/run_pipeline.py train        # train + evaluate the classifier
    python scripts/run_pipeline.py scan         # rank today's pump candidates
    python scripts/run_pipeline.py all          # everything, in order
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pump_predictor.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pipeline")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["universe", "download", "dataset", "train", "scan", "all"])
    ap.add_argument("--config", default=None, help="path to config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.step in ("universe", "all"):
        from pump_predictor.universe import build_universe
        build_universe(cfg)

    if args.step in ("download", "all"):
        from pump_predictor.market_data import download_ohlcv
        from pump_predictor.universe import load_universe
        download_ohlcv(cfg, load_universe(cfg))

    if args.step in ("dataset", "all"):
        from pump_predictor.dataset import build_dataset
        build_dataset(cfg)

    if args.step in ("train", "all"):
        import pandas as pd
        from pump_predictor.model import train_model
        data = pd.read_parquet(cfg.paths.dataset_file)
        metrics = train_model(data, cfg)
        print(json.dumps(metrics, indent=2))

    if args.step in ("scan", "all"):
        from pump_predictor.scan import run_scan
        top = run_scan(cfg)
        pd_opt = __import__("pandas")
        with pd_opt.option_context("display.width", 160, "display.max_columns", 30):
            print(top.to_string(index=False))
        out = Path(cfg.paths.data_dir) / "scan_latest.csv"
        top.to_csv(out, index=False)
        log.info("Scan saved -> %s", out)


if __name__ == "__main__":
    main()

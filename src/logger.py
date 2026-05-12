import csv
import json
from dataclasses import asdict
from pathlib import Path

import gymnasium
import numpy as np
import torch

from utils import get_git_hash


class Logger:
    """Logs training metrics to CSV and run metadata to JSON."""

    def __init__(self, run_dir: Path, config: object) -> None:
        self.run_dir = run_dir
        self.csv_path = run_dir / "metrics.csv"
        self.meta_path = run_dir / "metadata.json"
        self._csv_file = None
        self._writer = None

        self._save_metadata(config)

    def _save_metadata(self, config: object) -> None:
        """Save run metadata: config, library versions, git hash."""
        meta = {
            "config": asdict(config),
            "versions": {
                "torch": torch.__version__,
                "gymnasium": gymnasium.__version__,
                "numpy": np.__version__,
            },
            "git_hash": get_git_hash(),
            "device": str(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"),
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def log(self, metrics: dict) -> None:
        """Append a row of metrics to the CSV file."""
        if self._csv_file is None:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=list(metrics.keys()))
            self._writer.writeheader()

        self._writer.writerow(metrics)
        self._csv_file.flush()

    def close(self) -> None:
        """Close the CSV file."""
        if self._csv_file is not None:
            self._csv_file.close()

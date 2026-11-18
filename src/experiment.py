import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from .dataset import chronological_split, limit_training_frames, normalization, open_cache, split_summary

PROJECT = Path(__file__).resolve().parents[1]
WIDTH = 64
LATENT = 48
PARAMETERS = 9_793_460


@dataclass
class Experiment:
    data: Path
    run: Path = PROJECT / "runs" / "sparc_seed42"
    validation_data: Path | None = None
    test_data: Path | None = None
    seed: int = 42
    steps: int = 10_000
    batch_size: int = 4
    train_frames: int | None = None
    validation_start: str | None = None
    test_start: str | None = None
    gap_hours: int = 24
    validation_interval: int = 1_000
    checkpoint_interval: int = 500
    log_interval: int = 25
    validation_frames: int = 32
    test_frames: int = 64
    patches_per_frame: int = 4
    workers: int = 0
    learning_rate: float = 2e-4
    aux_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"
    amp: bool = True
    lambda_base_rate: float = 0.05
    lambda_full_rate: float = 0.025
    full_distortion_weight: float = 0.75
    detail_sparsity_weight: float = 2e-4
    spectral_weight: float = 0.002
    precipitation_weight: float = 0.02

    def __post_init__(self):
        self.data = Path(self.data).resolve()
        self.run = Path(self.run).resolve()
        for name in ("validation_data", "test_data"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, Path(value).resolve())

    def data_paths(self):
        if self.validation_data is None:
            return dict.fromkeys(("train", "validation", "test"), self.data)
        paths = {"train": self.data, "validation": self.validation_data}
        if self.test_data is not None:
            paths["test"] = self.test_data
        return paths


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def atomic_write(path, write):
    path = Path(path)
    with NamedTemporaryFile(dir=path.parent, prefix="." + path.name, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, data):
    atomic_write(path, lambda target: target.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    ))


def write_stats(path, stats):
    def save(target):
        with target.open("wb") as handle:
            np.savez(handle, mean=stats[0], std=stats[1])
    atomic_write(path, save)


def load_stats(path):
    with np.load(path, allow_pickle=False) as values:
        return values["mean"].astype(np.float32), values["std"].astype(np.float32)


def inspect_experiment(config):
    root, train_times, train_info = open_cache(config.data)
    paths = config.data_paths()
    times, stores = {"train": train_times}, {"train": train_info}
    for name, path in paths.items():
        if name == "train":
            continue
        if path == config.data:
            times[name], stores[name] = train_times, train_info
        else:
            _, times[name], stores[name] = open_cache(path)
    if config.validation_data is None:
        splits = chronological_split(train_times, config.seed, config.train_frames, config.gap_hours,
                                     config.validation_start, config.test_start)
    else:
        splits = {name: np.argsort(axis) for name, axis in times.items()}
        splits["train"] = limit_training_frames(splits["train"], config.train_frames, config.seed)
    cache_id = hashlib.sha256("".join(store["cache_id"] for store in stores.values()).encode()).hexdigest()
    return root, times, {"cache_id": cache_id, "stores": stores}, splits


def prepare_run(config, inspected, resume=False):
    root, times, info, splits = inspected
    run = config.run
    experiment_file = run / "experiment.json"
    statistics_file = run / "normalization.npz"
    settings = asdict(config)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in settings.items()}
    manifest = {
        "settings": settings,
        "model": {"width": WIDTH, "latent": LATENT, "patch_size": 64, "parameters": PARAMETERS},
        "dataset": info,
        "split": split_summary(times, splits),
        "split_time_indices": {name: indices.tolist() for name, indices in splits.items()},
        "split_timestamps_ns": {name: times[name][indices].tolist() for name, indices in splits.items()},
        "versions": {name: version(name) for name in ("torch", "compressai", "numpy", "zarr", "numcodecs")},
    }

    if resume:
        stats = load_stats(statistics_file)
    else:
        run.mkdir(parents=True)
        stats = normalization(root, splits["train"])
        write_stats(statistics_file, stats)
    write_json(experiment_file, manifest)
    return stats

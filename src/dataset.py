import hashlib
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .model import CHANNEL_NAMES

STATIC_NAMES = ["land_sea_mask", "normalized_orography", "sin_latitude", "cos_latitude"]
SST_INDEX = CHANNEL_NAMES.index("sst")
CACHE_NAME = "era5_28ch_0p5.zarr"
QUICK_CACHE_NAME = "era5_28ch_0p5_quick4.zarr"
DIRS = (Path("data"), Path("data/hackathons/NuclearITHackSchool/mifi_hack_airi"))
PARTITION_NAMES = {
    "train": "era5_28ch_0p5_train128.zarr",
    "validation": "era5_28ch_0p5_validation2020_32.zarr",
    "test": "era5_28ch_0p5_test2021_64.zarr",
}


def partition_paths(project_dir):
    if project_dir.parent.name == "hackathons":
        data_dir = project_dir.parents[1] / "data" / "hackathons" / project_dir.name
    else:
        data_dir = project_dir / "data"
    return {name: data_dir / filename for name, filename in PARTITION_NAMES.items()}


def default_data(project_dir, smoke=False):
    names = (QUICK_CACHE_NAME, CACHE_NAME) if smoke else (CACHE_NAME,)
    for name in names:
        for base in (project_dir, *project_dir.parents):
            for directory in DIRS:
                path = base / directory / name
                if (path / ".zgroup").is_file():
                    return path
    return partition_paths(project_dir)["train"] if smoke else project_dir / "data" / CACHE_NAME


def timestamp(ns):
    return np.datetime_as_string(np.datetime64(int(ns), "ns"), unit="s") + "Z"


def open_cache(path):
    root = zarr.open_group(str(Path(path)), mode="r")
    weather = root["weather"]
    times = root["time_ns"][:]
    lat, lon = root["latitude"][:], root["longitude"][:]
    ocean = root["ocean_weight"][:]
    gaps = np.diff(np.sort(times)) / 3.6e12
    signature = hashlib.sha256(times.tobytes())
    signature.update(str(weather.shape).encode())
    info = {
        "cache_id": signature.hexdigest(),
        "source_uri": root.attrs.get("source_uri"),
        "sampling": root.attrs.get("sampling"),
        "weather_shape": list(weather.shape),
        "weather_chunks": list(weather.chunks),
        "earliest": timestamp(times.min()),
        "latest": timestamp(times.max()),
        "gap_hours_min_median_max": [float(gaps.min()), float(np.median(gaps)), float(gaps.max())] if gaps.size else [],
        "latitude_endpoints": [float(lat[0]), float(lat[-1])],
        "longitude_endpoints": [float(lon[0]), float(lon[-1])],
        "grid_degrees": float(lon[1] - lon[0]),
        "channel_names": root.attrs.get("channel_names"),
        "static_names": root.attrs.get("static_names"),
        "sst_ocean_fraction": float((ocean > 0.5).mean()),
    }
    return root, times, info


def chronological_split(times, seed, train_frames=None, gap_hours=24,
                        validation_start=None, test_start=None):
    order = np.argsort(times)
    ordered = times[order]
    if validation_start is None and test_start is None:
        test_count = max(1, len(order) // 10)
        first, second = len(order) - 2 * test_count, len(order) - test_count
    else:
        cutoffs = np.array([validation_start, test_start], dtype="datetime64[ns]").astype(np.int64)
        first, second = np.searchsorted(ordered, cutoffs)
    gap_ns = int(gap_hours * 3_600_000_000_000)
    train = order[:np.searchsorted(ordered[:first], ordered[first] - gap_ns, side="right")]
    valid = order[first:first + np.searchsorted(ordered[first:second], ordered[second] - gap_ns, side="right")]
    test = order[second:]
    train = limit_training_frames(train, train_frames, seed)
    return {"train": train, "validation": valid, "test": test}


def limit_training_frames(indices, limit, seed):
    if limit is None:
        return indices
    positions = np.random.default_rng(seed).choice(len(indices), limit, replace=False)
    return indices[np.sort(positions)]


def split_summary(times, splits):
    return {
        name: {"frames": len(indices), "start": timestamp(times[name][indices[0]]),
               "end": timestamp(times[name][indices[-1]])}
        for name, indices in splits.items()
    }


def normalization(root, indices):
    weather = root["weather"]
    ocean = root["ocean_weight"][:] > 0.5
    total = np.zeros(28, dtype=np.float64)
    squares = np.zeros_like(total)
    count = np.full(28, len(indices) * ocean.size, dtype=np.int64)
    count[SST_INDEX] = len(indices) * np.count_nonzero(ocean)
    for index in tqdm(indices, desc="Нормализация train", unit="кадр"):
        values = weather[int(index)].reshape(28, -1)
        values[SST_INDEX, ~ocean.ravel()] = 0
        total += values.sum(axis=1, dtype=np.float64)
        squares += np.einsum("ij,ij->i", values, values, dtype=np.float64)
    mean = total / count
    variance = squares / count - mean**2
    std = np.sqrt(np.maximum(variance, 0)).clip(min=1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def wrapped(array, prefix, rows, col, size):
    width = array.shape[-1]
    end = col + size
    if end <= width:
        return np.asarray(array[(*prefix, rows, slice(col, end))])
    first = np.asarray(array[(*prefix, rows, slice(col, width))])
    second = np.asarray(array[(*prefix, rows, slice(0, end - width))])
    return np.concatenate((first, second), axis=-1)


class PatchDataset(Dataset):
    def __init__(self, path, indices, stats, seed, *, length=None, offset=0,
                 patches_per_frame=1, include_raw=False):
        self.path = str(path)
        self.indices = np.asarray(indices)
        self.mean, self.std = stats
        self.seed = seed
        self.length = length
        self.offset = offset
        self.patches_per_frame = patches_per_frame
        self.include_raw = include_raw
        self.patch = 64
        self.root = None

    def __getstate__(self):
        loaded = {"root", "latitude", "static", "ocean", "weather"}
        return {**{key: value for key, value in self.__dict__.items() if key not in loaded}, "root": None}

    def __len__(self):
        return self.length if self.length is not None else len(self.indices) * self.patches_per_frame

    def __getitem__(self, index):
        index = range(len(self))[index]
        if self.root is None:
            self.root = zarr.open_group(self.path, mode="r")
            self.latitude = self.root["latitude"][:]
            self.static = self.root["static"][:]
            self.ocean = self.root["ocean_weight"][:]
            self.weather = self.root["weather"]

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.offset + index]))
        if self.length is not None:
            time_index = int(rng.choice(self.indices))
            area_quantile = rng.random()
        else:
            frame, band = divmod(index, self.patches_per_frame)
            time_index = int(self.indices[frame])
            area_quantile = (band + rng.random()) / self.patches_per_frame
        latitude = np.rad2deg(np.arcsin(2 * area_quantile - 1))
        center = int(np.argmin(np.abs(self.latitude - latitude)))
        row = int(np.clip(center - self.patch // 2, 0, self.weather.shape[-2] - self.patch))
        col = int(rng.integers(self.weather.shape[-1]))
        rows = slice(row, row + self.patch)
        raw = wrapped(self.weather, (time_index, slice(None)), rows, col, self.patch)
        static = wrapped(self.static, (slice(None),), rows, col, self.patch)
        ocean = wrapped(self.ocean, (), rows, col, self.patch)
        x = (raw - self.mean[:, None, None]) / self.std[:, None, None]
        x[SST_INDEX] = np.where(ocean > 0.5, x[SST_INDEX], 0.0)

        result = {
            "x": torch.from_numpy(x),
            "static": torch.from_numpy(static.copy()),
            "ocean": torch.from_numpy(ocean.copy()),
            "latitude": torch.from_numpy(self.latitude[rows].copy()),
            "time_index": time_index,
            "row": row,
            "col": col,
        }
        if self.include_raw:
            result["raw"] = torch.from_numpy(raw)
        return result


def evaluation_frames(indices, limit):
    positions = np.linspace(0, len(indices) - 1, min(len(indices), limit), dtype=int)
    return indices[positions]

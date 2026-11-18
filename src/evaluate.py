import argparse
import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from .dataset import SST_INDEX, PatchDataset, evaluation_frames, normalization, open_cache, timestamp
from .experiment import LATENT, WIDTH, Experiment, inspect_experiment, load_stats, write_json
from .model import CHANNEL_NAMES, SPARCCodec

PHYSICAL = [
    ("°C", 1.0, -273.15), ("hPa", 0.01, 0.0), ("m/s", 1.0, 0.0), ("m/s", 1.0, 0.0),
    ("mm / 6 h", 1000.0, 0.0), ("°C", 1.0, -273.15), ("kg/m²", 1.0, 0.0), ("fraction", 1.0, 0.0),
    *[("°C", 1.0, -273.15)] * 4,
    *[("m/s", 1.0, 0.0)] * 8,
    *[("m²/s²", 1.0, 0.0)] * 4,
    *[("g/kg", 1000.0, 0.0)] * 4,
]
MODES = ("base", "full")


def stream_header(stream):
    length = struct.unpack(">I", stream[5:9])[0]
    return json.loads(stream[9:9 + length])


@torch.inference_mode()
def codec_roundtrip(model, x, static):
    streams, reconstructions = {}, {}
    for mode in MODES:
        streams[mode] = model.encode(x, static, detail=(mode == "full"))
        reconstructions[mode] = model.decode(streams[mode], static)
    return streams, reconstructions


def plot_history(history, destination):
    figure, (loss_axis, rate_axis) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for key, label, color in (("train", "Train", "#31668a"), ("validation", "Validation", "#c75a34")):
        steps = [row["step"] for row in history[key]]
        style = {"alpha": 0.65, "linewidth": 1} if key == "train" else {"marker": ".", "linewidth": 1.5}
        loss_axis.plot(steps, [row["loss"] for row in history[key]], label=label, color=color, **style)
        rate_axis.plot(steps, [row["full_bpv"] for row in history[key]], label=label, color=color, **style)
    loss_axis.set(title="SPARC-ERA5 training", ylabel="Rate-distortion loss")
    rate_axis.set(xlabel="Optimizer step", ylabel="Estimated full rate\n(bits / value)")
    for axis in (loss_axis, rate_axis):
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def plot_comparison(physical, latitude, longitude, destination):
    fields = [
        (0, "Температура\nна высоте 2 м", "coolwarm"),
        (1, "Давление\nна уровне моря", "viridis"),
        (2, "Зональный ветер\nна высоте 10 м", "RdBu_r"),
        (3, "Меридиональный\nветер, 10 м", "RdBu_r"),
        (4, "Осадки за 6 ч", "Blues"),
    ]
    figure, axes = plt.subplots(len(fields), 3, figsize=(9.5, 11), layout="constrained")
    dy, dx = np.diff(latitude).mean(), np.diff(longitude).mean()
    extent = (longitude[0] - dx / 2, longitude[-1] + dx / 2,
              latitude[0] - dy / 2, latitude[-1] + dy / 2)
    for row, (channel, name, palette) in enumerate(fields):
        lower = min(float(array[channel].min()) for array in physical.values())
        upper = max(float(array[channel].max()) for array in physical.values())
        if channel in {2, 3}:
            upper = max(abs(lower), abs(upper))
            lower = -upper
        if lower == upper:
            upper += max(abs(lower), 1.0) * 1e-5
        for col, (mode, array) in enumerate(physical.items()):
            axis = axes[row, col]
            image = axis.imshow(array[channel], origin="lower", extent=extent, cmap=palette,
                                vmin=lower, vmax=upper, interpolation="nearest")
            axis.set(xticks=[], yticks=[])
            if row == 0:
                axis.set_title(mode)
            if col == 0:
                axis.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=10)
        colorbar = figure.colorbar(image, ax=axes[row, :], shrink=0.85)
        colorbar.set_label(PHYSICAL[channel][0])
    figure.savefig(destination, dpi=155, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)


def plot_rate_quality(metrics, destination, split="test"):
    figure, axis = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    for name, color in zip(MODES, ("#31668a", "#c75a34")):
        mode = metrics[name]
        axis.scatter(mode["bits_per_value"], mode["normalized_rmse"], s=55, color=color, label=name.title())
    axis.set(title=f"ERA5 patches ({split})", xlabel="Measured rate (bits / value)",
             ylabel="Normalized RMSE")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def evaluate_run(run, data=None, split="test"):
    run = Path(run)
    experiment = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    config = experiment["settings"]
    saved_indices = experiment["split_time_indices"].get(split)
    data_path = data or config[f"{split}_data"]
    if data_path is None and saved_indices is not None:
        data_path = config["data"]
    root, times, info = open_cache(data_path)
    indices = np.array(saved_indices) if saved_indices is not None else np.argsort(times)
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    stats = load_stats(run / "normalization.npz")
    seed = config["seed"] + (1 if split == "validation" else 2)
    dataset = PatchDataset(data_path, evaluation_frames(indices, config[f"{split}_frames"]),
                           stats, seed, patches_per_frame=config["patches_per_frame"],
                           include_raw=True)
    return evaluate_dataset(run, checkpoint, dataset, root, times, stats, split, "best.pt", info)


def evaluate_checkpoint(run, checkpoint_path, train_data, validation_data):
    config = Experiment(data=train_data, validation_data=validation_data, run=run, device="cpu")
    root, times, info = open_cache(config.validation_data)
    train_root, _, _, splits = inspect_experiment(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    stats = normalization(train_root, splits["train"])
    indices = evaluation_frames(splits["validation"], config.validation_frames)
    dataset = PatchDataset(config.validation_data, indices, stats, config.seed + 1,
                           patches_per_frame=config.patches_per_frame, include_raw=True)
    config.run.mkdir(parents=True, exist_ok=True)
    return evaluate_dataset(config.run, checkpoint, dataset, root, times, stats,
                            "validation", Path(checkpoint_path).name, info)


def evaluate_dataset(run, checkpoint, dataset, root, times, stats, split, checkpoint_name, cache_info):
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 4))
        model = SPARCCodec(width=WIDTH, latent=LATENT).eval()
        model.load_state_dict(checkpoint["model"])
        model.update()
        metrics = measure_test(run, model, dataset, root, times, stats, split)
    finally:
        torch.set_num_threads(previous_threads)
    metrics["checkpoint"] = {"file": checkpoint_name, "step": checkpoint["step"],
                             "validation_loss": checkpoint["best_loss"]}
    metrics[f"{split}_cache"] = {"path": dataset.path, **cache_info}
    metrics["split"] = split
    write_json(run / f"{split}_metrics.json", metrics)
    plot_rate_quality(metrics["modes"], run / "rate_quality.png", split)
    return metrics


def measure_test(run, model, dataset, root, times, stats, split="test"):
    counts = np.zeros(len(CHANNEL_NAMES), dtype=np.int64)
    normalized_sse = {name: np.zeros_like(counts, dtype=np.float64) for name in MODES}
    physical_sse = {name: np.zeros_like(counts, dtype=np.float64) for name in MODES}
    total_bytes = dict.fromkeys(MODES, 0)
    payload_bytes = dict.fromkeys(MODES, 0)
    raw_bytes = 0
    mean, std = (values.astype(np.float64)[:, None, None] for values in stats)
    scale = np.array([entry[1] for entry in PHYSICAL])[:, None, None]
    offset = np.array([entry[2] for entry in PHYSICAL])[:, None, None]
    best_rain = -np.inf
    lat, lon = root["latitude"][:], root["longitude"][:]

    for index in tqdm(range(len(dataset)), desc=f"{split} encode/decode", unit="patch"):
        sample = dataset[index]
        raw = sample["raw"].numpy()
        x = sample["x"].numpy()
        ocean = sample["ocean"].numpy() > 0.5
        raw_bytes += raw.nbytes
        counts += ocean.size
        counts[SST_INDEX] += ocean.sum() - ocean.size
        streams, decoded = codec_roundtrip(model, sample["x"][None], sample["static"][None])
        predictions = {name: decoded[name][0].numpy().astype(np.float64) for name in MODES}
        for name, prediction in predictions.items():
            total_bytes[name] += len(streams[name])
            header = stream_header(streams[name])
            payload_bytes[name] += sum(header["lengths"])
            normal_error = prediction - x
            native_error = prediction * std + mean - raw
            normal_error[SST_INDEX, ~ocean] = 0
            native_error[SST_INDEX, ~ocean] = 0
            normalized_sse[name] += np.square(normal_error).sum(axis=(1, 2))
            physical_sse[name] += np.square(native_error * scale).sum(axis=(1, 2))
        rain = float(raw[4].max())
        if rain > best_rain:
            best_rain = rain
            example = (sample, streams, {"Original": raw * scale + offset,
                        "Base": (predictions["base"] * std + mean) * scale + offset,
                        "Full": (predictions["full"] * std + mean) * scale + offset})

    sample, streams, physical = example
    lat_patch = lat[sample["row"]:sample["row"] + dataset.patch]
    lon_patch = lon[sample["col"]] + np.arange(dataset.patch) * (lon[1] - lon[0])
    label = timestamp(times[sample["time_index"]])
    plot_comparison(physical, lat_patch, lon_patch, run / "reconstructions.png")
    for name, stream in streams.items():
        (run / f"example_{name}.bin").write_bytes(stream)
    np.savez(run / "example_static.npz", static=sample["static"].numpy()[None],
             latitude=lat_patch, longitude=lon_patch)

    modes = {}
    for name in MODES:
        modes[name] = {
            "bitstream_bytes": total_bytes[name],
            "entropy_payload_bytes": payload_bytes[name],
            "header_bytes": total_bytes[name] - payload_bytes[name],
            "compression_ratio": raw_bytes / total_bytes[name],
            "bits_per_value": 8 * total_bytes[name] / (raw_bytes // 4),
            "normalized_rmse": float(np.sqrt(normalized_sse[name].sum() / counts.sum())),
            "rmse_by_channel": {
                channel: {"unit": PHYSICAL[i][0], "valid_points": int(counts[i]),
                          "value": float(np.sqrt(physical_sse[name][i] / counts[i])) if counts[i] else None,
                          "normalized": float(np.sqrt(normalized_sse[name][i] / counts[i])) if counts[i] else None}
                for i, channel in enumerate(CHANNEL_NAMES)
            },
        }
    return {
        "frames": len(dataset.indices), "timestamps_utc": [timestamp(times[index]) for index in dataset.indices],
        "patches": len(dataset), "patch_size": dataset.patch, "raw_float32_weather_bytes": raw_bytes,
        "size_definition": "Sum of 28 × H × W × 4 weather bytes / sum of complete SPRC1 streams. "
                           "Includes magic, header and entropy payload, with base included in full. "
                           "Excludes static inputs, normalization, weights, timestamps and Zarr storage.",
        "rmse_definition": "Pooled pointwise RMSE on decoded patches; SST uses ocean_weight > 0.5. "
                           "Normalized global RMSE pools valid values across channels. Physical RMSE is per channel.",
        "modes": modes,
        "incremental_detail_entropy_bytes": payload_bytes["full"] - payload_bytes["base"],
        "example": {"timestamp_utc": label, "latitude_start_index": sample["row"],
                    "longitude_start_index": sample["col"],
                    "selection": "Patch with maximum original 6 h precipitation"},
    }


def main():
    parser = argparse.ArgumentParser(description="Оценка SPARC-ERA5 через encode/decode")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--data", type=Path, help="Путь к Zarr для оценки")
    parser.add_argument("--checkpoint", type=Path, help="Оценить опубликованный checkpoint на validation")
    parser.add_argument("--train-data", type=Path, help="Train Zarr для восстановления нормализации")
    args = parser.parse_args()
    if args.checkpoint:
        result = evaluate_checkpoint(args.run, args.checkpoint, args.train_data, args.data)
    else:
        result = evaluate_run(args.run, args.data, args.split)
    for name, mode in result["modes"].items():
        print(f"{name}: {mode['bitstream_bytes']:,} bytes, CR={mode['compression_ratio']:.2f}, "
              f"normalized RMSE={mode['normalized_rmse']:.4f}")


if __name__ == "__main__":
    main()

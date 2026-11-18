import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import PatchDataset, default_data, evaluation_frames, normalization, open_cache, partition_paths
from .experiment import (
    LATENT, PROJECT, WIDTH, Experiment, atomic_write, device_for,
    inspect_experiment, prepare_run, write_json,
)
from .model import SPARCCodec, loss_fn


def init_model(seed, device):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return SPARCCodec(width=WIDTH, latent=LATENT).to(device)


def build_loaders(config, splits, stats, device, start_step=0):
    train = PatchDataset(config.data, splits["train"], stats, config.seed,
                         length=(config.steps - start_step) * config.batch_size,
                         offset=start_step * config.batch_size)
    validation = PatchDataset(
        config.data_paths()["validation"], evaluation_frames(splits["validation"], config.validation_frames),
        stats, config.seed + 1, patches_per_frame=config.patches_per_frame,
    )
    options = {
        "batch_size": config.batch_size,
        "num_workers": config.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.workers > 0,
    }
    return (
        DataLoader(train, **options, generator=torch.Generator().manual_seed(config.seed)),
        DataLoader(validation, **options, generator=torch.Generator().manual_seed(config.seed + 1)),
    )


def to_device(batch, device):
    return {name: batch[name].to(device, non_blocking=True) for name in ("x", "static", "latitude", "ocean")}


def forward_loss(model, batch, config, amp):
    with torch.amp.autocast(batch["x"].device.type, enabled=amp):
        output = model(batch["x"], batch["static"])
    return loss_fn({name: value.float() for name, value in output.items()}, batch, config)


@torch.no_grad()
def validate(model, loader, config, device):
    model.eval()
    totals = {}
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        _, metrics = forward_loss(model, batch, config, amp=False)
        size = len(batch["x"])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + size * value
        count += size
    return {key: value / count for key, value in totals.items()}


def save_best(run, state):
    best = {key: state[key] for key in ("model", "step", "best_loss", "dataset_id", "width", "latent")}
    atomic_write(run / "best.pt", lambda path: torch.save(best, path))


def train_model(config, model, splits, stats, dataset_id, device, resume=False):
    named = list(model.named_parameters())
    main_parameters = [value for name, value in named if not name.endswith(".quantiles")]
    aux_parameters = [value for name, value in named if name.endswith(".quantiles")]
    optimizer = torch.optim.AdamW(main_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    auxiliary = torch.optim.Adam(aux_parameters, lr=config.aux_learning_rate)
    amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start = 0
    best_loss = float("inf")
    best_step = None
    history = {"train": [], "validation": []}
    model.train()
    if resume:
        state = torch.load(config.run / "latest.pt", map_location="cpu", weights_only=True)
        start = state["step"]
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        auxiliary.load_state_dict(state["auxiliary"])
        if amp and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        best_loss, best_step, history = state["best_loss"], state["best_step"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        if best_step == start:
            save_best(config.run, state)
        write_json(config.run / "history.json", history)
        del state

    train_loader, validation_loader = build_loaders(config, splits, stats, device, start)
    last_validation = history["validation"][-1] if history["validation"] else None
    with tqdm(train_loader, total=config.steps, initial=start, desc="Train SPARC-ERA5", unit="step") as progress:
        for step, batch in enumerate(progress, start=start + 1):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            auxiliary.zero_grad(set_to_none=True)
            loss, metrics = forward_loss(model, batch, config, amp)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(main_parameters, 1.0)
            scaler.step(optimizer)
            auxiliary.zero_grad(set_to_none=True)
            aux_loss = model.aux_loss()
            aux_loss.backward()
            auxiliary.step()
            scaler.update()

            if step == 1 or step % config.log_interval == 0 or step == config.steps:
                history["train"].append({"step": step, **metrics})
            validated = step % config.validation_interval == 0 or step == config.steps
            if validated:
                last_validation = {"step": step, **validate(model, validation_loader, config, device)}
                history["validation"].append(last_validation)
                if last_validation["loss"] < best_loss:
                    best_loss, best_step = last_validation["loss"], step
                model.train()

            postfix = {"loss": f"{metrics['loss']:.4f}", "train bpv": f"{metrics['full_bpv']:.4f}"}
            if last_validation is not None:
                postfix.update({"val loss": f"{last_validation['loss']:.4f}",
                                "val bpv": f"{last_validation['full_bpv']:.4f}"})
            progress.set_postfix(postfix, refresh=False)

            if validated or step % config.checkpoint_interval == 0:
                state = {
                    "step": step, "best_loss": best_loss, "best_step": best_step, "history": history,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "auxiliary": auxiliary.state_dict(), "scaler": scaler.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    "dataset_id": dataset_id, "width": WIDTH, "latent": LATENT,
                }
                atomic_write(config.run / "latest.pt", lambda path: torch.save(state, path))
                if best_step == step:
                    save_best(config.run, state)
                write_json(config.run / "history.json", history)
    return history


def smoke_test(data, seed=42):
    from .evaluate import codec_roundtrip

    root, times, info = open_cache(data)
    indices = np.argsort(times)[:2]
    stats = normalization(root, indices)
    dataset = PatchDataset(data, indices, stats, seed, length=12)
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    config = Experiment(data, seed=seed, device="cpu")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 4))
        model = init_model(seed, torch.device("cpu"))
        loss, metrics = forward_loss(model, batch, config, amp=False)
        loss.backward()
        model.eval()
        model.update()
        streams, _ = codec_roundtrip(model, batch["x"], batch["static"])
        return {
            "available_frames": len(times), "shape": info["weather_shape"],
            "patch_shape": list(batch["x"].shape), "smoke_loss": metrics["loss"],
            "bytes": {name: len(stream) for name, stream in streams.items()},
        }
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description="Обучение SPARC-ERA5 на локальном Zarr")
    parser.add_argument("--data", type=str, help="Путь к локальному Zarr v2")
    parser.add_argument("--validation-data", help="Путь к отдельному validation Zarr")
    parser.add_argument("--test-data", help="Путь к отдельному test Zarr")
    parser.add_argument("--run", type=str, default=PROJECT / "runs" / "sparc_seed42")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--inspect", action="store_true")
    for name in (
        "seed", "steps", "batch_size", "train_frames", "gap_hours", "workers",
        "validation_interval", "checkpoint_interval", "validation_frames", "test_frames",
        "patches_per_frame",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(Experiment, name))
    for name in ("validation_start", "test_start"):
        parser.add_argument("--" + name.replace("_", "-"))
    parser.add_argument("--learning-rate", type=float, default=Experiment.learning_rate)
    args = vars(parser.parse_args())
    smoke, inspect, resume = (args.pop(name) for name in ("smoke", "inspect", "resume"))
    paths = partition_paths(PROJECT)
    if not (smoke or inspect) and not args["data"] and paths["train"].is_dir():
        for key, name in (("data", "train"), ("validation_data", "validation")):
            args[key] = args[key] or paths[name]
        if args["test_data"] is None:
            previous_experiment = Path(args["run"]) / "experiment.json"
            if resume and previous_experiment.is_file():
                previous = json.loads(previous_experiment.read_text(encoding="utf-8"))
                args["test_data"] = previous["settings"]["test_data"]
            elif (paths["test"] / ".zgroup").is_file():
                args["test_data"] = paths["test"]
    args["data"] = args["data"] or default_data(PROJECT, smoke=smoke or inspect)
    config = Experiment(**args)
    if inspect:
        print(json.dumps(open_cache(config.data)[2], indent=2, ensure_ascii=False))
        return
    if smoke:
        print(json.dumps(smoke_test(config.data, config.seed), indent=2, ensure_ascii=False))
        return

    from .evaluate import evaluate_run, plot_history

    inspected = inspect_experiment(config)
    _, _, info, splits = inspected
    device = device_for(config.device)
    stats = prepare_run(config, inspected, resume)
    print(f"{device}, AMP={config.amp and device.type == 'cuda'}, "
          + ", ".join(f"{name}: {len(indices)}" for name, indices in splits.items()))
    model = init_model(config.seed, device)
    history = train_model(config, model, splits, stats, info["cache_id"], device, resume)
    del model
    plot_history(history, config.run / "training.png")
    evaluate_run(config.run, split="test" if "test" in splits else "validation")


if __name__ == "__main__":
    main()

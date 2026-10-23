"""Train one member on a shared split; save the exact config and code revision."""

import argparse
import csv
import json
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import SteelDataset, load_annotations
from .metrics import DiceMeter, segmentation_loss
from .models import make_model


def read_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    parent = config.pop("extends", None)
    return {**(read_config(path.parent / parent) if parent else {}), **config}


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


@torch.inference_mode()
def validate(model, loader, config):
    model.eval()
    meter = DiceMeter()
    classification_rows = []
    for images, masks, presence, names in loader:
        images = images.cuda(non_blocking=True)
        with torch.autocast("cuda", enabled=config["amp"]):
            probability = model(images).sigmoid()
        if config["task"] == "segmentation":
            meter.update(probability > config["threshold"], masks.cuda(non_blocking=True), names)
        else:
            for name, p, y in zip(names, probability.float().cpu().tolist(), presence.tolist()):
                classification_rows.append({"image": name, "probability": p, "target": y})
    if config["task"] == "segmentation":
        result = meter.summary()
        return result, meter.rows, result["nonempty_dice"]
    probs = np.array([row["probability"] for row in classification_rows])
    truth = np.array([row["target"] for row in classification_rows], dtype=bool)
    prediction = probs > config["threshold"]
    tp, fp, fn = (prediction & truth).sum(0), (prediction & ~truth).sum(0), (~prediction & truth).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    result = {"macro_f1": float(f1.mean()), "class_f1": f1.tolist(),
              "tp": tp.tolist(), "fp": fp.tolist(), "fn": fn.tolist()}
    return result, classification_rows, result["macro_f1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Only for a smoke run")
    parser.add_argument("--max-val-images", type=int, default=0, help="Only for a smoke run")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if not torch.cuda.is_available():
        raise RuntimeError("Run training on Legion with CUDA")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    split = json.loads(Path(config["split"]).read_text())
    labels = load_annotations(config["data_dir"])
    if set(split["train"]) & set(split["val"]):
        raise ValueError("Train and validation overlap")
    if set(split["train"]) | set(split["val"]) != set(labels):
        raise ValueError("Split does not cover the available training images")
    val_names = split["val"][:args.max_val_images] if args.max_val_images else split["val"]
    train_data = SteelDataset(config["data_dir"], split["train"], labels, True, config["crop_width"], config["brightness_contrast"])
    val_data = SteelDataset(config["data_dir"], val_names, labels)
    generator = torch.Generator().manual_seed(config["seed"])
    options = dict(num_workers=config["workers"], pin_memory=True, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, generator=generator, **options)
    val_loader = DataLoader(val_data, batch_size=config["val_batch_size"], shuffle=False, **options)
    model = make_model(config, pretrained=not args.resume).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    start, best, stale = 0, -1.0, 0
    if args.resume:
        # Only load checkpoints produced by this project into the same run directory.
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        previous = state["config"]
        if {k: v for k, v in config.items() if k != "epochs"} != {k: v for k, v in previous.items() if k != "epochs"}:
            raise ValueError("Resume config changed; create a new run instead")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best, stale = state["epoch"] + 1, state["best"], state["stale"]
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_state"])
        torch.set_rng_state(state["torch_state"])
        torch.cuda.set_rng_state_all(state["cuda_states"])
        generator.set_state(state["loader_state"])
    else:
        config_text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        (output / "config.yaml").write_text(config_text)
        shutil.copyfile(config["split"], output / "split.json")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    metadata = {
        "time": datetime.now(timezone(timedelta(hours=8))).isoformat(), "git_commit": commit,
        "git_dirty": bool(dirty), "config": config, "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda, "resume": args.resume,
        "train_images": len(train_data), "val_images": len(val_data),
        "max_train_batches": args.max_train_batches, "smoke_run": bool(args.max_train_batches or args.max_val_images),
        "checkpoint_selection": "nonempty_dice" if config["task"] == "segmentation" else "macro_f1",
    }
    with (output / "runs.jsonl").open("a") as stream:
        stream.write(json.dumps(metadata) + "\n")
    emit("start", **metadata)
    for epoch in range(start, config["epochs"]):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        loss_sum, samples = 0.0, 0
        for step, (images, masks, presence, _) in enumerate(train_loader):
            if args.max_train_batches and step >= args.max_train_batches:
                break
            images = images.cuda(non_blocking=True)
            target = (masks if config["task"] == "segmentation" else presence).cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=config["amp"]):
                logits = model(images)
                if config["task"] == "segmentation":
                    loss = segmentation_loss(logits, target, config["positive_weights"], config["bce_weight"])
                else:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, step {step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(images)
            samples += len(images)
            if (step + 1) % 100 == 0:
                emit("progress", epoch=epoch + 1, step=step + 1, steps=len(train_loader),
                     loss=loss_sum / samples, seconds=round(time.perf_counter() - began, 1))
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - began
        val_start = time.perf_counter()
        metrics, rows, score = validate(model, val_loader, config)
        scheduler.step(score)
        improved = score > best
        best, stale = (score, 0) if improved else (best, stale + 1)
        record = {"epoch": epoch + 1, "loss": loss_sum / max(samples, 1), "train_seconds": train_seconds,
                  "val_seconds": time.perf_counter() - val_start,
                  "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                  "learning_rate": optimizer.param_groups[0]["lr"], "metrics": metrics, "best_score": best}
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "best": best,
                 "stale": stale, "config": config, "git_commit": commit,
                 "random_state": random.getstate(), "numpy_state": np.random.get_state(),
                 "torch_state": torch.get_rng_state(), "cuda_states": torch.cuda.get_rng_state_all(),
                 "loader_state": generator.get_state()}
        torch.save(state, output / "last.tmp")
        (output / "last.tmp").replace(output / "last.pt")
        if improved:
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch + 1, "git_commit": commit}, output / "best.pt")
            (output / "best_metrics.json").write_text(json.dumps(record, indent=2))
            with (output / "best_validation.jsonl").open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        emit("epoch", **record)
        if stale >= config["patience"]:
            emit("early_stop", epoch=epoch + 1)
            break
    emit("complete", best_score=best, output=str(output))


if __name__ == "__main__":
    main()

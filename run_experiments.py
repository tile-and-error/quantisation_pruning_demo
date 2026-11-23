#!/usr/bin/env python3
"""Run from this folder: .venv/bin/python run_experiments.py

Everything written here; ../diss/datasets is read-only. No experiment framework.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache"))
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import timm
import methods as M


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def preprocess(x):
    # Tiny images are 64x64; use the SAME resize + normalisation for every variant.
    x = F.interpolate(x.float() / 255, (224, 224), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    return (x - mean) / std


def load_data(args, out):
    rng = np.random.default_rng(args.seed)
    root = args.data.resolve()
    manifest, arrays, labels = {}, {}, {}
    for split, count in (("train", args.train_per_class + args.calibration_per_class),
                         ("test", args.samples_per_class)):
        paths, ys, ims = [], [], []
        dirs = sorted(p for p in (root / split).iterdir() if p.is_dir())
        if [p.name for p in dirs] != [f"{i:03d}" for i in range(200)]:
            raise ValueError("Expected Tiny ImageNet's numeric class folders 000..199")
        for label, folder in enumerate(dirs):
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
            if len(files) < count: raise ValueError(f"Not enough images: {folder}")
            for index in rng.choice(len(files), size=count, replace=False):
                path = files[index]
                with Image.open(path) as image:
                    ims.append(np.array(image.convert("RGB"), copy=True))
                paths.append(str(path.relative_to(root))); ys.append(label)
        arrays[split] = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).contiguous()
        labels[split] = torch.tensor(ys)
        manifest[split] = paths
    # Within each class the final images are reserved for calibration / short fine-tuning.
    stride = args.train_per_class + args.calibration_per_class
    mask = torch.arange(len(labels["train"])) % stride < args.train_per_class
    train_x, train_y = arrays["train"][mask], labels["train"][mask]
    cal_x, cal_y = arrays["train"][~mask], labels["train"][~mask]
    # Shuffle once, so small calibration/training batches mix classes.
    idx = torch.from_numpy(rng.permutation(len(cal_y)))
    cal_x, cal_y = cal_x[idx], cal_y[idx]
    manifest["head_train"] = [p for p, keep in zip(manifest["train"], mask) if keep]
    manifest["calibration"] = [manifest["train"][i] for i in torch.where(~mask)[0][idx].tolist()]
    assert not set(manifest["head_train"]) & set(manifest["calibration"])
    assert not set(manifest["train"]) & set(manifest["test"])
    (out / "samples.json").write_text(json.dumps(manifest, indent=2))
    print(f"Data: {len(train_y)} head-training, {len(cal_y)} calibration, {len(labels['test'])} evaluation images", flush=True)
    return train_x, train_y, preprocess(cal_x), cal_y, arrays["test"], labels["test"]


def build_base(kind, train_x, train_y, args, out):
    torch.manual_seed(args.seed)
    if kind == "resnet18":
        model = torchvision.models.quantization.resnet18(weights=None, quantize=False)
        weight_path = ROOT / "cache/resnet18-f37072fd.pth"
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        dim, head_name = model.fc.in_features, "fc"
    else:
        model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
        weight_path = ROOT / "cache/deit_tiny_patch16_224-a1311bcf.pth"
        model.load_state_dict(torch.load(weight_path, weights_only=True)["model"])
        # Explicit attention in ALL variants, including baseline, for a fair graph comparison.
        for block in model.blocks: block.attn.fused_attn = False
        dim, head_name = model.head.in_features, "head"
    model.eval()
    setattr(model, head_name, nn.Identity())
    features = []
    t = time.perf_counter()
    with torch.inference_mode():
        for i, batch in enumerate(M.batches(train_x, args.batch_size)):
            features.append(model(preprocess(batch)))
            if i % 25 == 0: print(f"{kind}: extracting head-training features {i*args.batch_size}/{len(train_x)}", flush=True)
    features = torch.cat(features).clone()
    # Standardise frozen features, then fold the standardisation into the final Linear.
    mean, std = features.mean(0), features.std(0).clamp_min(1e-4)
    z = (features - mean) / std
    head = nn.Linear(dim, 200)
    optim = torch.optim.Adam(head.parameters(), lr=.01, weight_decay=.01)
    losses = []
    for i in range(args.head_steps):
        loss = F.cross_entropy(head(z), train_y)
        optim.zero_grad(); loss.backward(); optim.step()
        if i % 20 == 0: losses.append({"step": i, "loss": loss.item()})
    with torch.no_grad():
        head.weight.div_(std[None]); head.bias.sub_(head.weight @ mean)
    setattr(model, head_name, head)
    info = {"head_steps": args.head_steps, "training_images": len(train_y), "seconds": time.perf_counter()-t,
            "loss_trace": losses, "pretrained_weights": str(weight_path), "sha256": digest(weight_path)}
    (out / f"{kind}_head_training.json").write_text(json.dumps(info, indent=2))
    return model.eval()


@torch.inference_mode()
def evaluate(model, x, y, batch_size):
    logits = torch.cat([model(preprocess(b)) for b in M.batches(x, batch_size)])
    if not torch.isfinite(logits).all(): raise RuntimeError("Non-finite predictions")
    pred = logits.topk(5, dim=1).indices
    correct = pred[:, 0].eq(y)
    n = len(y); p = correct.float().mean().item()
    # Wilson descriptive interval; small stratified subset and shared pretrained data limit inference.
    z = 1.96; denom = 1+z*z/n
    centre = (p+z*z/(2*n))/denom
    half = z*((p*(1-p)/n+z*z/(4*n*n))**.5)/denom
    return {"top1_pct": 100*p, "top5_pct": 100*pred.eq(y[:, None]).any(1).float().mean().item(),
            "top1_wilson_low_pct": 100*(centre-half), "top1_wilson_high_pct": 100*(centre+half),
            "evaluation_images": n}, pred[:, 0]


def tensor_stats(model):
    tensors = []
    def walk(v):
        if torch.is_tensor(v): tensors.append(v)
        elif isinstance(v, dict):
            for value in v.values(): walk(value)
        elif isinstance(v, (tuple, list)):
            for value in v: walk(value)
    walk(model.state_dict())
    weights = [x for x in tensors if x.ndim >= 2]
    count = sum(x.numel() for x in weights)
    zeros = sum((x.int_repr() == 0).sum().item() if x.is_quantized else (x == 0).sum().item() for x in weights)
    qparam_bytes = 0
    for x in tensors:
        if x.is_quantized and x.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
            qparam_bytes += x.q_per_channel_scales().numel()*8 + x.q_per_channel_zero_points().numel()*8
    return {"state_tensor_mib": (sum(x.numel()*x.element_size() for x in tensors)+qparam_bytes)/2**20,
            "state_tensor_elements": sum(x.numel() for x in tensors),
            "weight_zero_pct": 100*zeros/max(1, count),
            "quantized_module_count": sum("quantized" in type(m).__module__ and hasattr(m, "weight") for m in model.modules())}


def bench_worker(args):
    """Fresh process: excludes transforms, calibration, evaluation arrays and other models."""
    import psutil
    import threading
    torch.backends.quantized.engine = "fbgemm"
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    process = psutil.Process()
    before = process.memory_info().rss
    model = torch.jit.load(args.artifact, map_location="cpu").eval()
    x = torch.load(args.bench_input, weights_only=True)
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(.002): peak[0] = max(peak[0], process.memory_info().rss)
    sampler = threading.Thread(target=sample, daemon=True); sampler.start()
    try:
        with torch.inference_mode():
            for _ in range(10): model(x[:1])
            timings = []
            for _ in range(args.repeats):
                start = time.perf_counter_ns(); model(x[:1]); timings.append((time.perf_counter_ns()-start)/1e6)
            batch_times = []
            for _ in range(3): model(x)
            for _ in range(10):
                start = time.perf_counter(); model(x); batch_times.append(time.perf_counter()-start)
        peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        stop.set(); sampler.join()
    result = {"latency_median_ms": float(np.median(timings)), "latency_p95_ms": float(np.percentile(timings, 95)),
              "batch_throughput_images_s": len(x)/float(np.median(batch_times)), "throughput_batch_size": len(x),
              "process_peak_rss_mib": peak[0]/2**20, "process_import_rss_mib": before/2**20,
              "rss_above_import_mib": (peak[0]-before)/2**20, "latencies_ms": timings}
    Path(args.bench_result).write_text(json.dumps(result, indent=2))


def main(args):
    torch.backends.quantized.engine = "fbgemm"
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True); (out / "models").mkdir(exist_ok=True)
    (out / "details").mkdir(exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update({"torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
                   "python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
                   "input_shape": [3,224,224], "dataset": "Tiny ImageNet / zh-plus/tiny-imagenet",
                   "normalisation_mean": [.485,.456,.406], "normalisation_std": [.229,.224,.225],
                   "quantized_backend": "fbgemm", "source_sha256": {p.name:digest(p) for p in (ROOT/"methods.py", ROOT/"run_experiments.py")}})
    config_path = out / "config.json"
    if (out / "results.json").exists() and config_path.exists():
        old = json.loads(config_path.read_text())
        for key in ("seed", "samples_per_class", "train_per_class", "calibration_per_class", "head_steps", "steps", "data", "threads", "batch_size", "repeats", "source_sha256"):
            if old.get(key) != config.get(key): raise ValueError(f"Output already contains a different run ({key}); choose --output")
    config_path.write_text(json.dumps(config, indent=2))
    train_x, train_y, cal_x, cal_y, test_x, test_y = load_data(args, out)
    torch.save(cal_x[:args.batch_size].clone(), out / "bench_input.pt")
    rows = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else []
    # Each tuple is (name, description); transforms below are deliberately explicit.
    cnn = [("baseline", "FP32"), ("random30", "random global weights 30%"), ("random50", "random global weights 50%"),
           ("magnitude30", "magnitude global weights 30%"), ("magnitude50", "magnitude global weights 50%"),
           ("structural25", "L1 hidden channels 25%"), ("structural50", "L1 hidden channels 50%"),
           ("slim_dense", "short BN-L1 fine-tune; dense control"), ("slimming25", "BN-L1 + hidden channels 25%"),
           ("slimming50", "BN-L1 + hidden channels 50%"), ("int8", "static INT8 PTQ"),
           ("random30_int8", "random 30% + static INT8"), ("magnitude30_int8", "magnitude 30% + static INT8"),
           ("structural25_int8", "hidden channels 25% + static INT8"), ("slimming25_int8", "BN-L1 channels 25% + static INT8")]
    vit = [("baseline", "FP32"), ("w4", "W4A32 nearest rounding simulation"), ("uniform", "W4 + selected A4 uniform simulation"),
           ("twin", "W4 + selected A4 two-region simulation"), ("scale", "LN/Linear equalisation + W4/selected A4 simulation"),
           ("qat", "short STE QAT on ViT; Q-ViT-inspired only"), ("block", "W4A32 block bias reconstruction demo"),
           ("adaround", "W4A32 learned rounding demo"), ("int8", "real dynamic INT8 Linear kernels"),
           ("structural25", "physically prune MLP hidden units 25%"), ("structural25_int8", "MLP 25% + dynamic INT8")]
    vit += [(f"structural25_{q}", f"MLP 25% + {q} demo") for q in ("twin", "scale", "qat", "block", "adaround")]
    wanted = set(args.only.split(",")) if args.only else None
    for kind, variants in (("resnet18", cnn), ("deit_tiny", vit)):
        todo = [(name, desc) for name,desc in variants if (wanted is None or f"{kind}/{name}" in wanted)
                and not any(r["model"] == kind and r["variant"] == name for r in rows)]
        if not todo: continue
        base_path = out / "models" / f"{kind}_source.pt"
        base = torch.load(base_path, weights_only=False) if base_path.exists() else build_base(kind, train_x, train_y, args, out)
        if not base_path.exists(): torch.save(base, base_path)
        slim = None
        for name, description in todo:
            print(f"\nRunning {kind}/{name}: {description}", flush=True)
            torch.manual_seed(args.seed)
            model, trace = copy.deepcopy(base), []
            start = time.perf_counter()
            if kind == "resnet18":
                if name.startswith("random"): M.unstructured(model, .5 if "50" in name else .3, random=True)
                if name.startswith("magnitude"): M.unstructured(model, .5 if "50" in name else .3)
                if name.startswith("structural"): M.structural(model, .5 if "50" in name else .25)
                if name.startswith("slim"):
                    if slim is None:
                        slim, trace = M.slim_train(copy.deepcopy(base), cal_x, cal_y, args.steps)
                        (out / "details/slimming_training.json").write_text(json.dumps(trace, indent=2))
                    model = copy.deepcopy(slim)
                    if name != "slim_dense": M.structural(model, .5 if "50" in name else .25, slimming=True)
                if name.endswith("int8"): model = M.static_int8(model, cal_x)
            else:
                if name.startswith("structural25"): M.prune_mlp(model)
                q = name.removeprefix("structural25_")
                if q == "w4": M.weight_round(model)
                if q == "scale":
                    model, error = M.scale_reparameterise(model, cal_x)
                    trace.append({"fp32_rewrite_max_error": error})
                if q in ("uniform", "twin", "scale", "qat"):
                    model, qt = M.fake_vit(model, cal_x, twin=q=="twin", qat=q=="qat", steps=args.steps, targets=cal_y)
                    trace.extend(qt)
                if q in ("block", "adaround"): model, trace = M.reconstruct(model, cal_x, args.steps, adaptive=q=="adaround")
                if q == "int8": model = M.dynamic_int8(model)
            transform_seconds = time.perf_counter()-start
            model.eval()
            metrics, predictions = evaluate(model, test_x, test_y, args.batch_size)
            artifact = out / "models" / f"{kind}_{name}.pt"
            # TorchScript avoids the eager quantised Conv pickle bug and is standalone.
            with torch.inference_mode():
                exported = torch.jit.trace(model, cal_x[:1])
                torch.jit.save(exported, artifact)
            del exported
            # Reload parity catches broken surgery/quantisation serialization.
            with torch.inference_mode():
                probe = cal_x[:2]
                expected = model(probe)
                reloaded = torch.jit.load(artifact).eval()
                reload_error = (expected-reloaded(probe)).abs().max().item()
                if reload_error > 1e-5: raise RuntimeError(f"Reload mismatch: {reload_error}")
            del reloaded
            detail = out / "details" / f"{kind}_{name}_benchmark.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--bench-worker", "--artifact", str(artifact),
                       "--bench-input", str(out/"bench_input.pt"), "--bench-result", str(detail),
                       "--threads", str(args.threads), "--repeats", str(args.repeats)]
            subprocess.run(command, check=True)
            bench = json.loads(detail.read_text()); bench.pop("latencies_ms")
            stats = tensor_stats(model)
            if name.endswith("int8") and not stats["quantized_module_count"]: raise RuntimeError("No quantised modules converted")
            row = {"model": kind, "variant": name, "description": description, **metrics, **bench, **stats,
                   "artifact_mib": artifact.stat().st_size/2**20, "transform_seconds": transform_seconds,
                   "reload_max_error": reload_error, "artifact": str(artifact.relative_to(ROOT)),
                   "execution": "real INT8" if name.endswith("int8") else "FP32 simulation" if kind=="deit_tiny" and name not in ("baseline", "structural25") else "FP32"}
            rows.append(row)
            (out / "results.json").write_text(json.dumps(rows, indent=2))
            (out / "details" / f"{kind}_{name}_transform.json").write_text(json.dumps(trace, indent=2))
            np.save(out / "details" / f"{kind}_{name}_predictions.npy", predictions.numpy())
            print(f"top1={row['top1_pct']:.1f}% top5={row['top5_pct']:.1f}% latency={row['latency_median_ms']:.2f}ms tensors={row['state_tensor_mib']:.2f}MiB artifact={row['artifact_mib']:.2f}MiB", flush=True)
            del model; gc.collect()
        del base, slim; gc.collect()
    from make_report import report
    report(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT.parent/"diss/datasets/tiny-imagenet")
    p.add_argument("--output", default="results")
    p.add_argument("--seed", type=int, default=407)
    p.add_argument("--samples-per-class", type=int, default=5)
    p.add_argument("--train-per-class", type=int, default=10)
    p.add_argument("--calibration-per-class", type=int, default=1)
    p.add_argument("--head-steps", type=int, default=200)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--only", help="comma separated model/variant names; completed variants resume automatically")
    p.add_argument("--bench-worker", action="store_true")
    for name in ("artifact", "bench-input", "bench-result"): p.add_argument(f"--{name}")
    args = p.parse_args()
    if args.bench_worker: bench_worker(args)
    else:
        for key in ("samples_per_class", "train_per_class", "calibration_per_class", "head_steps", "steps", "threads", "batch_size", "repeats"):
            if getattr(args, key) < 1: p.error(f"--{key.replace('_','-')} must be positive")
        main(args)

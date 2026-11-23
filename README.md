# Quick pruning + quantisation experiments

Start with **[the measured meeting report](results/MEETING_REPORT.md)**, **[the overview plot](results/overview.png)** and **[the CSV](results/results.csv)**.

Everything lives in this directory. The dataset in `../diss/datasets/tiny-imagenet` is read only. No changes to the `diss` project are needed. The linked [PR #4](https://github.com/tile-and-error/imagenet-classification/pull/4) provides project context; this deliberately stays a standalone Python experiment, without adding its pipeline abstractions.

## Run

The isolated `.venv`, pretrained checkpoints, and dependencies are already installed here:

```bash
cd /dcs/23/u5534387/git/diss_temp
.venv/bin/python -u run_experiments.py
```

For a **quick live meeting demo** using the saved models, with no training or downloads:

```bash
.venv/bin/python demo.py --image-index 0
```

This loads FP32, magnitude-pruned and real INT8 models, runs a saved held-out image, and prints its predicted class, a short live timing, and the already measured accuracy and model bytes. Pick another evaluation image with `--image-index` (0–999 for the supplied run).

This runs all 31 configurations and writes `results/`. Completed configurations resume automatically. The first run fits a small linear classifier head for each pretrained backbone. Subsequent runs reuse its saved baseline. Use a new output directory if changing the configuration or code.

```bash
# A small end-to-end check of every technique:
.venv/bin/python -u run_experiments.py --output quick_run \
  --train-per-class 2 --samples-per-class 1 --head-steps 40 --steps 2 --repeats 5

# Run selected comparisons:
.venv/bin/python -u run_experiments.py --output selected_run \
  --only resnet18/baseline,resnet18/structural25,resnet18/int8,resnet18/structural25_int8

# Regenerate the charts/tables from existing results:
.venv/bin/python make_report.py results

# Show the two quantisation grids on actual model activations:
.venv/bin/python quantisation_plot.py

# Mathematical checks for masking vs physical surgery and quantiser levels:
.venv/bin/python verify_transforms.py
```

`--data` accepts another directory with this Tiny ImageNet layout: `train/000..199` and `test/000..199`. This script expects all 200 classes. The local `test` directory contains the downloaded labelled validation split. It is not the unlabelled competition test set. Full ImageNet-1k was **not** present, so the supplied run is explicitly a Tiny ImageNet pilot.

## What to open

| File | Purpose |
|---|---|
| `results/MEETING_REPORT.md` | Measured results, useful comparisons, method descriptions and limits |
| `results/overview.png` / `.svg` | Accuracy, latency and tensor size for all experiments |
| `results/tradeoffs.png` | Accuracy against latency, with model bytes shown by point area |
| `results/memory.png` | Model tensor bytes against fresh-process inference RSS |
| `results/quantisation_regions.png` | Uniform vs twin grids on actual ViT activations |
| `results/results.csv` / `.json` | All measured metrics, including top-1/top-5, p95 latency, throughput, model bytes, memory and converted module counts |
| `results/samples.json` / `config.json` | Exact image paths, split separation, seed, software versions, source hashes and settings |
| `results/details/` | Raw timing samples, transform diagnostics, predicted classes and reload checks |
| `results/models/` | Standalone TorchScript executable models, plus fitted FP32 source models |
| `results/*head_training.json` | Original checkpoint SHA-256, fitted-head training budget and loss trace |
| `experiments.log` / `vit_experiments.log` | Progress/output of the supplied CNN and ViT suites |

## The simple code

- `run_experiments.py`: sample data, fit linear heads, explicitly apply each transform, evaluate, save/reload, benchmark in a separate process.
- `methods.py`: random/magnitude masks, residual-safe channel surgery, short network slimming, true INT8 conversions, low-bit simulations, scale equalisation, short ViT QAT, block bias reconstruction and learned rounding.
- `make_report.py`: CSV, static Matplotlib figures and meeting notes.
- `quantisation_plot.py`: show the actual grids and errors on representative calibration activations.
- `verify_transforms.py`: small correctness checks for physical pruning and quantisation.
- `demo.py`: quick live inference with the saved executable models.

The default run uses ImageNet-pretrained ResNet18 and DeiT-Tiny, each with a **new fitted 200-class linear head** trained on 10 local training images per class. It evaluates on 5 validation images per class (1,000 total). Another 1 training image per class is reserved for calibration and short adaptation. These are separate files. No validation images are used to tune transformations.

The 200-class heads are important: Tiny ImageNet's class IDs do not match pretrained 1,000-class outputs, and 18 classes lack a direct 1,000-class output match. `cache/labels.json` records the exact local label order, names and available ImageNet-1k index matches. Both backbones were pretrained on ImageNet-1k, from which Tiny ImageNet derives, so original pretraining exposure cannot be excluded; this is a controlled compression demonstration rather than a new generalisation benchmark.

## Keep the claims straight

**True INT8 uses real CPU quantised kernels and packed model weights.** CNN static PTQ includes convolutions; ViT dynamic INT8 converts Linear layers. Report the coverage, because other operations remain FP32.

**All 4-bit experiments are simulations using FP32 tensors and kernels.** They measure effects on accuracy and any simulation overhead, but do not demonstrate INT4 memory reduction or speedup. Each advanced method is a short, explicitly documented demonstration of the core idea. The QAT example is Q-ViT-inspired, the equalisation is related to RepQ-ViT, the bias correction is BRECQ-inspired, and learned rounding is AdaRound-inspired; none is a full paper reproduction.

Model tensor memory, actual model file bytes and total fresh-process RSS are reported separately. RSS includes interpreter/runtime and workspace allocations, sampled every 2 ms; it is not an allocator-exact peak activation count. Inference timing excludes data loading and preprocessing. Both backbones use the same 224x224 input and one CPU thread. DeiT uses explicit attention in every variant, including its baseline, to make activation simulation comparisons more consistent. Executable artifacts are exported with `torch.jit.trace` and reloaded with `torch.jit.load`; the source model files are trusted local Python pickles. Traces are validated at batch sizes 1, 2 and the benchmark batch size for the fixed 224x224 input shape.

## References and environment

Copies of the supplied guides are in `references/`:

- [Pruning guide](references/pruning_guide_cs407.md)
- [Quantisation guide](references/quantisation_from_fundamentals_to_computational_pathology.md)

`cache/` contains original pretrained weights, fetched dataset metadata, the label table and a copy of the PR branch's plan. Python 3.11, CPU PyTorch 2.5.1 and torchvision 0.20.1 are pinned here to use their established eager convolutional quantisation route. All installed packages are recorded in `requirements-lock.txt`.

To rebuild the environment, without changing `diss`:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv venv --python /usr/bin/python3.11 .venv
UV_CACHE_DIR="$PWD/.uv-cache" uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -r requirements-lock.txt
```

The pretrained files already exist; rerunning experiments needs no network. Randomness is seeded (407 by default). Timing is subject to host load; one seed and short training budgets are sufficient for a meeting demo, not a conclusive ranking of research methods.

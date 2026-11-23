"""Regenerate CSV, meeting notes, and static figures from measured results."""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def report(out):
    out = Path(out)
    rows = json.loads((out / "results.json").read_text())
    config = json.loads((out / "config.json").read_text())
    if not (out/"quantisation_regions.png").exists() and (out/"models/deit_tiny_source.pt").exists():
        from quantisation_plot import main as plot_regions
        plot_regions(out)
    for model in {r["model"] for r in rows}:
        baseline = next((r for r in rows if r["model"]==model and r["variant"]=="baseline"), None)
        if baseline:
            for r in rows:
                if r["model"] == model:
                    r["accuracy_change_pp"] = r["top1_pct"]-baseline["top1_pct"]
                    r["speedup"] = baseline["latency_median_ms"]/r["latency_median_ms"]
                    r["tensor_size_ratio"] = baseline["state_tensor_mib"]/r["state_tensor_mib"]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    models = [m for m in ("resnet18", "deit_tiny") if any(r["model"]==m for r in rows)]
    fig, axs = plt.subplots(len(models), 3, figsize=(19, max(6, len(models)*6)), squeeze=False)
    for i, model in enumerate(models):
        group = [r for r in rows if r["model"]==model]
        y = np.arange(len(group))
        colors = ["#159c79" if r["execution"]=="real INT8" else "#efad4d" if r["execution"]=="FP32 simulation" else "#427cbb" for r in group]
        for j,(metric, label) in enumerate((("top1_pct", "Top-1 accuracy (%)"), ("latency_median_ms", "Batch-1 CPU latency (ms)"), ("state_tensor_mib", "Model tensor storage (MiB)"))):
            ax = axs[i,j]
            ax.barh(y, [r[metric] for r in group], color=colors)
            ax.set_yticks(y, [r["variant"] for r in group] if j==0 else [])
            ax.invert_yaxis(); ax.set_xlabel(label); ax.set_title(model.replace("_", " ")+" · "+label)
            ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
            for yy,r in zip(y,group):
                ax.text(r[metric], yy, f" {r[metric]:.1f}", va="center", fontsize=8)
            ax.set_xlim(0, max(r[metric] for r in group)*1.16)
    fig.suptitle(f"Measured Tiny ImageNet pilot · {config['samples_per_class']*200} evaluation images · CPU / {config['threads']} thread(s)\nBlue: FP32   Green: real INT8 kernels   Orange: simulated low-bit values in FP32", fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(out / "overview.png", dpi=160); fig.savefig(out / "overview.svg"); plt.close(fig)

    fig, axs = plt.subplots(1,len(models),figsize=(8*len(models),6),squeeze=False)
    for ax,model in zip(axs[0],models):
        group=[r for r in rows if r["model"]==model]
        labelled = {"baseline", "magnitude30", "int8", "structural25_int8"} if model=="resnet18" else {"baseline", "w4", "uniform", "twin", "block", "adaround", "int8", "structural25_int8"}
        for r in group:
            color = "#159c79" if r["execution"]=="real INT8" else "#efad4d" if r["execution"]=="FP32 simulation" else "#427cbb"
            ax.scatter(r["latency_median_ms"],r["top1_pct"],s=max(30,r["state_tensor_mib"]*8),c=color,alpha=.75,edgecolors="white")
            if r["variant"] in labelled:
                offset=(5,-14) if r["variant"] in ("baseline","w4","int8") else (5,6)
                ax.annotate(r["variant"],(r["latency_median_ms"],r["top1_pct"]),xytext=offset,textcoords="offset points",fontsize=8)
        ax.set(title=model,xlabel="Batch-1 latency (ms; lower is better)",ylabel="Top-1 accuracy (%; higher is better)")
        ax.grid(alpha=.15);ax.margins(x=.22,y=.12)
    fig.suptitle("Accuracy / latency trade-off · bubble area represents model tensor bytes\nSelected labels; all configurations appear in the overview and tables")
    fig.tight_layout(); fig.savefig(out/"tradeoffs.png",dpi=180); plt.close(fig)

    fig, axs = plt.subplots(1,len(models),figsize=(8*len(models),7),squeeze=False)
    for ax,model in zip(axs[0],models):
        group=[r for r in rows if r["model"]==model]
        y=np.arange(len(group)); ax.barh(y-.18,[r["state_tensor_mib"] for r in group],.36,label="Model tensor storage")
        ax.barh(y+.18,[r["process_peak_rss_mib"] for r in group],.36,label="Fresh-process inference RSS")
        ax.set_yticks(y,[r["variant"] for r in group]);ax.invert_yaxis();ax.set(title=model,xlabel="MiB (includes interpreter/runtime for RSS)")
        ax.legend(loc="lower right");ax.grid(axis="x",alpha=.15)
    fig.suptitle("Memory: model bytes and total process memory answer different questions")
    fig.tight_layout(); fig.savefig(out/"memory.png",dpi=160);plt.close(fig)

    lines = ["# Measured pruning and quantisation pilot", "",
             f"{config['samples_per_class']*200} fixed evaluation images ({config['samples_per_class']} per class), all 200 Tiny ImageNet classes; "
             f"{config['train_per_class']*200} separate training images for a fitted linear head, {config['calibration_per_class']*200} separate training images for calibration / short adaptation.", "",
             "**These are measured pilot results, not ImageNet-1k benchmark scores or paper reproductions.** "
             "The only downloaded dataset was Tiny ImageNet. Both backbones were pretrained on ImageNet-1k; Tiny ImageNet derives from ImageNet, so prior image exposure cannot be excluded. "
             "The reported held-out split is held out from this experiment's head fitting, calibration and fine-tuning, not certified held out from original pretraining.", "",
             f"CPU: AMD EPYC 9654P; {config['threads']} intra-op thread(s), batch 1 latency after 10 warmups / {config['repeats']} repeats; "
             f"batch {config['batch_size']} throughput measured separately. All models receive identical 224x224 normalised images. "
             "Timing excludes image loading and preprocessing. Raw timings, sampled paths, reload checks and software versions are saved alongside this report.", "",
             "![All experiments](overview.png)", "", "![Trade-offs](tradeoffs.png)", "", "![Memory](memory.png)", "", "![Uniform and twin quantisation on real activations](quantisation_regions.png)", ""]
    lines += ["## Results", "", "Positive accuracy change means improvement; speedup >1 means faster than the same model's FP32 baseline. "
              "Small differences on this subset may be sampling or timing noise; see descriptive accuracy intervals in CSV.", "",
              "Matched comparisons: `random30` versus `magnitude30`; `w4` versus `block` / `adaround` (same W4A32 coverage); "
              "`uniform` versus `twin` / `scale` / `qat` (same W4 / selected A4 coverage). Comparing W4A32 reconstruction directly against selected-A4 QAT also changes activation precision. "
              "A collapse in the naive A4 softmax recipe can dominate the scale and short-QAT comparisons; this is not a ranking of the named papers.", ""]
    for model in models:
        lines += [f"### {model}", "", "| Experiment | Top-1 % | Top-5 % | Δ top-1 pp | Median ms | p95 ms | Speedup | Tensor MiB | File MiB | Peak RSS MiB | Execution |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in rows:
            if r["model"]!=model: continue
            lines.append(f"| {r['variant']} | {r['top1_pct']:.1f} | {r['top5_pct']:.1f} | {r.get('accuracy_change_pp',0):+.1f} | {r['latency_median_ms']:.2f} | {r['latency_p95_ms']:.2f} | {r.get('speedup',1):.2f}× | {r['state_tensor_mib']:.2f} | {r['artifact_mib']:.2f} | {r['process_peak_rss_mib']:.1f} | {r['execution']} |")
        lines += [""]
    lines += ["## Meeting observations from this run", ""]
    for model in models:
        group={r["variant"]:r for r in rows if r["model"]==model}
        if "baseline" not in group: continue
        b=group["baseline"]
        fastest=min(group.values(),key=lambda r:r["latency_median_ms"])
        lines += [f"- **{model}:** fastest measured variant was `{fastest['variant']}`: {fastest['latency_median_ms']:.2f} ms "
                  f"({fastest.get('speedup',1):.2f}× baseline), with {fastest['top1_pct']:.1f}% top-1 versus {b['top1_pct']:.1f}% baseline."]
        for a,c in (("random30","magnitude30"),("baseline","int8"),("structural25","structural25_int8"),("uniform","twin"),("w4","adaround"),("w4","block")):
            if a in group and c in group:
                x,y=group[a],group[c]
                lines += [f"- `{model}/{a}` → `{c}`: top-1 {x['top1_pct']:.1f}% → {y['top1_pct']:.1f}%; "
                          f"latency {x['latency_median_ms']:.2f} → {y['latency_median_ms']:.2f} ms; tensors {x['state_tensor_mib']:.2f} → {y['state_tensor_mib']:.2f} MiB."]
    region_path = out / "quantisation_region_errors.json"
    if region_path.exists():
        regions = json.loads(region_path.read_text())
        for name, numbers in regions.items():
            uniform, twin = numbers["Uniform A4"]["mse"], numbers["Twin A4"]["mse"]
            lines += [f"- First-block **{name}** on the plotted calibration batch: uniform A4 MSE {uniform:.3g}, twin A4 MSE {twin:.3g} "
                      f"({uniform/twin:.2f}× lower). This is a local numerical comparison, not a model accuracy guarantee."]
    lines += ["", "## What each experiment actually implements", "",
              "- Random / magnitude: global weight masks at 30% and 50%, materialised as dense zero tensors. Sparsity alone does not shrink this executable representation.",
              "- Structural CNN: rank hidden convolution filters by L1 norm; slice conv1 output, BN1, and conv2 input together. Residual widths stay intact. 25% hidden channels does not mean 25% total parameters.",
              f"- Network slimming: {config['steps']} supervised Adam steps with an L1 penalty on eligible BN scales, then rank those scales and physically shrink hidden channels. `slim_dense` is the adapted pre-pruning control. This short fine-tune cannot reproduce full sparsity training and recovery training from the paper.",
              "- Real INT8: ResNet uses fused convolution/BN/ReLU static PTQ with fbgemm, including quantised residual additions; DeiT uses dynamic INT8 Linear kernels. Other operations can remain FP32. Converted module counts and actual packed state bytes are in CSV.",
              "- ViT nearest W4A32: symmetric per-output-channel linear-weight rounding; patch convolution and other parameters remain FP32.",
              "- Twin: two uniform regions, using 75% of magnitude codes inside 1/8 of the calibrated max. Applied to MLP LN outputs, GELU outputs, and attention probabilities, alongside W4 linear weights. `uniform` uses the same selected A4 coverage with one region. This is a [PTQ4ViT-inspired](https://arxiv.org/abs/2111.12293) demonstration without its Hessian-guided calibration or exact search.",
              "- Scale reparameterisation: channel scales are moved algebraically from LayerNorm into the following Linear, with a numerical equivalence check before rounding. SmoothQuant-style channel equalisation motivated by [RepQ-ViT](https://openaccess.thecvf.com/content/ICCV2023/html/Li_RepQ-ViT_Scale_Reparameterization_for_Post-Training_Quantization_of_Vision_Transformers_ICCV_2023_paper.html); not its complete algorithm. Then the same W4 / selected A4 simulation is applied.",
              f"- [Q-ViT-inspired](https://arxiv.org/abs/2210.06707): {config['steps']} supervised STE fine-tuning steps on a ViT with W4 / selected A4 simulation. It demonstrates QAT on a transformer; it does not implement Q-ViT's information rectification module, distribution-guided distillation or full training recipe.",
              "- Block reconstruction: quantise each ViT block's Linear weights to W4 and optimise its Linear biases against the original local block outputs. Simple [BRECQ-inspired](https://arxiv.org/abs/2102.05426) MSE correction; not second-order BRECQ.",
              "- Adaptive rounding: optimise per-weight sigmoid rounding decisions against local block output MSE, encourage hard choices, then harden to adjacent grid points. Small [AdaRound-inspired](https://arxiv.org/abs/2004.10568) demonstration, not an optimised paper reproduction.",
              "- Combined experiments: pruning is applied first, then quantisation is calibrated or adapted on that pruned model. CNN combines every pruning family with real INT8. ViT combines physical MLP pruning with each advanced demo and dynamic INT8.", "",
              "**All W4 and A4 experiments execute FP32 kernels and store FP32 tensors. Their measured latency includes any explicit activation simulation overhead. They make no INT4 speed or memory saving claim.** "
              "INT8 files contain real packed weights. Checkpoint MiB includes module metadata; tensor MiB counts state tensors, including packed INT8 weights and scales. "
              "Peak RSS is sampled every 2 ms in a fresh process containing only the selected model and one benchmark batch; it includes Python, PyTorch, model and runtime workspaces. "
              "It is not allocator-exact peak activation memory. `rss_above_import_mib` is also recorded, but includes model loading and workspace allocations.", "",
              "## Limits and next useful checks", "",
              "This is one seed, a small stratified evaluation subset and very short adaptation budgets. No hyperparameters were selected on evaluation images. "
              "Pretrained features + a small fitted linear head do not equal a fully trained Tiny ImageNet model. Different backbones have different baselines: compare transformations within each backbone. "
              "The supplied Wilson intervals describe finite-sample uncertainty only; a paired multi-seed analysis is needed for small accuracy changes. "
              "A longer run should add sparse training/recovery, paired fine-tuning controls for QAT, a separate reconstruction validation split, full evaluation, and multiple seeds. "
              "Do not infer pathology performance from this dataset.", "",
              "Sources: the two supplied guides (see README), [project PR](https://github.com/tile-and-error/imagenet-classification/pull/4), "
              "[PTQ4ViT](https://arxiv.org/abs/2111.12293), [RepQ-ViT](https://openaccess.thecvf.com/content/ICCV2023/html/Li_RepQ-ViT_Scale_Reparameterization_for_Post-Training_Quantization_of_Vision_Transformers_ICCV_2023_paper.html), "
              "[Q-ViT](https://arxiv.org/abs/2210.06707), [BRECQ](https://arxiv.org/abs/2102.05426), [AdaRound](https://arxiv.org/abs/2004.10568).", ""]
    (out/"MEETING_REPORT.md").write_text("\n".join(lines))
    print(f"Report: {out/'MEETING_REPORT.md'}", flush=True)


if __name__=="__main__":
    import sys
    report(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/"results")

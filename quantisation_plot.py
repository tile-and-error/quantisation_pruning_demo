"""Show uniform/twin quantisers on actual first-block DeiT calibration activations."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import methods as M


def main(out=ROOT/"results"):
    torch.set_num_threads(1)
    model=torch.load(out/"models/deit_tiny_source.pt",weights_only=False).eval()
    x=torch.load(out/"bench_input.pt",weights_only=True)
    captured={}
    def save(key):
        def hook(m,args,y):captured[key]=y.detach().flatten().clone()
        return hook
    def probs(m,args):
        x=args[0];b,n,c=x.shape
        q,k,v=m.qkv(x).reshape(b,n,3,m.num_heads,c//m.num_heads).permute(2,0,3,1,4).unbind(0)
        captured["attention probabilities"]=((m.q_norm(q)*m.scale)@m.k_norm(k).transpose(-2,-1)).softmax(-1).flatten()
    handles=[model.blocks[0].mlp.act.register_forward_hook(save("GELU activations")),
             model.blocks[0].attn.register_forward_pre_hook(probs)]
    with torch.inference_mode():model(x)
    for h in handles:h.remove()
    fig,axs=plt.subplots(2,2,figsize=(12,8));metrics={}
    for i,(key,values) in enumerate(captured.items()):
        unsigned=key=="attention probabilities"
        maximum=values.abs().max().item()
        grid=torch.linspace(0 if unsigned else -maximum,maximum,4000)
        axs[i,0].hist(values.numpy(),bins=150,color="#506d8a",log=True)
        axs[i,0].set(title=key,xlabel="Calibration value",ylabel="Count (log scale)")
        metrics[key]={}
        for twin,color,label in ((False,"#427cbb","Uniform A4"),(True,"#e79632","Twin A4")):
            q=M.ActivationGrid(4,twin,unsigned)
            q(values);q.calibrating=False
            restored=q(values)
            mse=(restored-values).square().mean().item()
            metrics[key][label]={"mse":mse,"maximum":maximum,"inner_boundary":maximum/8,
                                "number_of_observed_codes":restored.unique().numel()}
            axs[i,1].plot(grid,q(grid),color=color,label=f"{label}; MSE={mse:.3g}",alpha=.8)
        axs[i,1].plot(grid,grid,"k--",linewidth=.7,alpha=.5)
        limit=maximum/3
        axs[i,1].set_xlim(0 if unsigned else -limit,limit)
        axs[i,1].set_ylim(0 if unsigned else -limit,limit)
        axs[i,1].set(title="Same 4-bit budget: finer steps near zero",xlabel="Original value",ylabel="Dequantised FP32 value")
        axs[i,1].legend(fontsize=8)
    fig.suptitle("Two regions with different step sizes · actual first-block DeiT calibration data\nNumerical illustration: these quantisers execute FP32 operations",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(out/"quantisation_regions.png",dpi=180);plt.close(fig)
    (out/"quantisation_region_errors.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":main()

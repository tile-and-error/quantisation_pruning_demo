"""Fast meeting demo: load saved models and run a real held-out image.

    .venv/bin/python demo.py --image-index 0

No training, calibration, or downloads. See results/MEETING_REPORT.md for accuracy.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


def main(index):
    torch.set_num_threads(1); torch.backends.quantized.engine="fbgemm"
    out=ROOT/"results"
    config=json.loads((out/"config.json").read_text())
    paths=json.loads((out/"samples.json").read_text())["test"]
    if not 0<=index<len(paths): raise ValueError(f"image-index must be 0..{len(paths)-1}")
    path=Path(config["data"])/paths[index]
    label=int(Path(paths[index]).parent.name)
    names=json.loads((ROOT/"cache/labels.json").read_text())
    with Image.open(path) as image:x=torch.from_numpy(np.array(image.convert("RGB"),copy=True)).permute(2,0,1)[None].float()/255
    x=F.interpolate(x,(224,224),mode="bilinear",align_corners=False,antialias=True)
    x=(x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
    rows=json.loads((out/"results.json").read_text())
    print(f"Image: {paths[index]}\nTrue class: {label:03d} — {names[label]['name']}\n")
    print(f"{'Model / variant':38} {'Prediction':30} {'Correct':8} {'Live ms':>8} {'Saved top1':>11} {'Tensor MiB':>11}")
    for row in rows:
        if row["variant"] not in ("baseline","magnitude30","int8","structural25_int8"):continue
        model=torch.jit.load(str(ROOT/row["artifact"])).eval()
        with torch.inference_mode():
            for _ in range(3):model(x)
            samples=[]
            for _ in range(10):
                start=time.perf_counter();logits=model(x);samples.append((time.perf_counter()-start)*1000)
        pred=logits.argmax(1).item()
        print(f"{row['model']+'/'+row['variant']:38} {names[pred]['name'][:28]:30} {str(pred==label):8} {np.median(samples):8.2f} {row['top1_pct']:10.1f}% {row['state_tensor_mib']:11.2f}")
        del model
    print("\nLive timing is a short illustration; use the report for warmed repeated benchmarks and held-out accuracy.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--image-index",type=int,default=0)
    main(parser.parse_args().image_index)

"""Meaningful small checks: surgery equivalence, zero budgets, adjacent rounding."""
import copy
import torch
from torch import nn
from torchvision.models.quantization import resnet18
import methods as M


def main():
    torch.set_num_threads(1); torch.manual_seed(407)
    base = resnet18(weights=None).eval()
    x = torch.randn(2, 3, 64, 64)
    for random in (False, True):
        m = M.unstructured(copy.deepcopy(base), .3, random=random)
        count = zeros = 0
        for layer in m.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                count += layer.weight.numel(); zeros += (layer.weight==0).sum().item()
        assert abs(zeros/count - .3) < 1/count
        assert sum(p.numel() for p in m.parameters())==sum(p.numel() for p in base.parameters())
    for slimming in (False, True):
        masked = copy.deepcopy(base)
        # Make gamma ranking nontrivial, also test slicing non-default BN buffers.
        for stage in (masked.layer1, masked.layer2, masked.layer3, masked.layer4):
            for b in stage:
                with torch.no_grad():
                    b.bn1.weight.uniform_(.2, 1); b.bn1.bias.uniform_(-.2, .2)
                    b.bn1.running_mean.uniform_(-.1, .1); b.bn1.running_var.uniform_(.8, 1.2)
        small = M.structural(copy.deepcopy(masked), .25, slimming=slimming)
        handles=[]
        for stage in (masked.layer1, masked.layer2, masked.layer3, masked.layer4):
            for block in stage:
                score=block.bn1.weight.abs() if slimming else block.conv1.weight.abs().flatten(1).sum(1)
                keep=score.topk(round(len(score)*.75)).indices
                mask=torch.zeros(len(score));mask[keep]=1
                handles.append(block.bn1.register_forward_hook(lambda m,a,out,mask=mask:out*mask[None,:,None,None]))
        with torch.inference_mode():
            assert torch.allclose(masked(x),small(x),atol=1e-5,rtol=1e-4), "Physical surgery differs from channel masking"
        for h in handles:h.remove()
        assert sum(p.numel() for p in small.parameters())<sum(p.numel() for p in masked.parameters())
        for stage in (small.layer1,small.layer2,small.layer3,small.layer4):
            for block in stage: assert block.conv1.out_channels==block.bn1.num_features==block.conv2.in_channels
    values=torch.linspace(-3,3,10001)
    for unsigned in (False,True):
        for twin in (False,True):
            q=M.ActivationGrid(4,twin,unsigned)
            v=values.abs() if unsigned else values
            q(v);q.calibrating=False
            result=q(v)
            assert result.unique().numel()<=16
            assert q(torch.zeros(1)).item()==0
            assert torch.isfinite(result).all()
    linear=nn.Linear(8,4)
    adaptive=M.AdaptiveLinear(linear)
    with torch.no_grad():
        hard=(adaptive.floor+(adaptive.alpha>=0).float())*adaptive.scale
        assert torch.allclose(hard,M.grid(linear.weight,4,axis=0),atol=1e-7)
    print("PASS: exact global zero budgets; dense shapes preserved; physical channel surgery matches masked outputs; residual dependencies intact; quantiser code budgets; initial adaptive hard rounding equals nearest rounding.")


if __name__=="__main__":main()

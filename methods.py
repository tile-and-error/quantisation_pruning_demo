"""Small, readable compression experiments. Paper-inspired demos, not reproductions."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import prune


def batches(x, size=16):
    return x.split(size)


@torch.no_grad()
def unstructured(model, amount, random=False):
    layers = [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(layers, pruning_method=prune.RandomUnstructured if random else prune.L1Unstructured,
                              amount=amount)
    for layer, name in layers:
        prune.remove(layer, name)  # Dense zero tensors; this does NOT shrink their storage.
    return model


@torch.no_grad()
def structural(model, amount, slimming=False):
    """Shrink only the hidden channels of residual blocks; keep skip interfaces intact."""
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in stage:
            score = block.bn1.weight.abs() if slimming else block.conv1.weight.abs().flatten(1).sum(1)
            keep = score.topk(max(1, round(len(score) * (1 - amount)))).indices.sort().values
            old1, oldbn, old2 = block.conv1, block.bn1, block.conv2
            conv1 = nn.Conv2d(old1.in_channels, len(keep), old1.kernel_size, old1.stride, old1.padding, bias=False)
            conv2 = nn.Conv2d(len(keep), old2.out_channels, old2.kernel_size, old2.stride, old2.padding, bias=False)
            bn = nn.BatchNorm2d(len(keep), eps=oldbn.eps, momentum=oldbn.momentum)
            conv1.weight.copy_(old1.weight[keep]); conv2.weight.copy_(old2.weight[:, keep])
            for attr in ("weight", "bias", "running_mean", "running_var"):
                getattr(bn, attr).copy_(getattr(oldbn, attr)[keep])
            bn.num_batches_tracked.copy_(oldbn.num_batches_tracked)
            block.conv1, block.bn1, block.conv2 = conv1, bn, conv2
    return model.eval()


@torch.no_grad()
def prune_mlp(model, amount=.25):
    for block in model.blocks:
        a, b = block.mlp.fc1, block.mlp.fc2
        score = a.weight.abs().sum(1) * b.weight.abs().sum(0)
        keep = score.topk(round(len(score) * (1 - amount))).indices.sort().values
        aa, bb = nn.Linear(a.in_features, len(keep)), nn.Linear(len(keep), b.out_features)
        aa.weight.copy_(a.weight[keep]); aa.bias.copy_(a.bias[keep])
        bb.weight.copy_(b.weight[:, keep]); bb.bias.copy_(b.bias)
        block.mlp.fc1, block.mlp.fc2 = aa, bb
    return model.eval()


def slim_train(model, x, y, steps=20, penalty=1e-3):
    """Short supervised fine-tune + L1 BN gamma penalty, with frozen BN statistics."""
    model.eval()
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    trace = []
    for i in range(steps):
        j = (i * 4) % len(x)
        logits = model(x[j:j+4])
        ce = F.cross_entropy(logits, y[j:j+4])
        l1 = sum(b.bn1.weight.abs().sum() for stage in (model.layer1, model.layer2, model.layer3, model.layer4) for b in stage)
        loss = ce + penalty * l1
        optim.zero_grad(); loss.backward(); optim.step()
        trace.append({"step": i, "ce": ce.item(), "gamma_l1": l1.item(), "loss": loss.item()})
    return model.eval(), trace


def static_int8(model, calibration):
    """Real quantised convolution, linear, residual-add kernels, using torchvision's ResNet."""
    from torch.ao.quantization import prepare, convert, get_default_qconfig
    torch.backends.quantized.engine = "fbgemm"
    model.eval(); model.fuse_model()
    model.qconfig = get_default_qconfig("fbgemm")
    prepare(model, inplace=True)
    with torch.inference_mode():
        for batch in batches(calibration):
            model(batch)
    convert(model, inplace=True)
    return model.eval()


def dynamic_int8(model):
    return torch.ao.quantization.quantize_dynamic(model.eval(), {nn.Linear}, dtype=torch.qint8)


def grid(x, bits, axis=None):
    qmax = 2 ** (bits - 1) - 1
    dims = tuple(i for i in range(x.ndim) if i != axis) if axis is not None else None
    scale = x.detach().abs().amax(dim=dims, keepdim=True).clamp_min(1e-8) / qmax
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


class ActivationGrid(nn.Module):
    """Calibrated signed uniform or two-region grid; all arithmetic remains FP32."""
    def __init__(self, bits=4, twin=False, unsigned=False):
        super().__init__()
        self.bits, self.twin, self.unsigned, self.calibrating = bits, twin, unsigned, True
        self.register_buffer("maximum", torch.tensor(1e-8))
        self.register_buffer("boundary", torch.tensor(1e-8))

    def forward(self, x):
        if self.calibrating:
            self.maximum.copy_(torch.maximum(self.maximum, x.detach().abs().max()))
            # Fixed fraction of the calibrated range: 75% of codes cover the centre.
            self.boundary.copy_(self.maximum / 8)
            return x
        qmax = 2 ** (self.bits - (0 if self.unsigned else 1)) - 1
        if not self.twin:
            s = self.maximum / qmax
            q = (x / s).round().clamp(0 if self.unsigned else -qmax, qmax) * s
        else:
            low_codes = int(qmax * .75)
            lo = self.boundary / low_codes
            hi = (self.maximum - self.boundary).clamp_min(1e-8) / (qmax - low_codes)
            a = x.abs()
            inside = (a / lo).round().clamp(0, low_codes) * lo
            outside = self.boundary + ((a - self.boundary) / hi).round().clamp(0, qmax-low_codes) * hi
            q = x.sign() * torch.where(a <= self.boundary, inside, outside)
        return x + (q - x).detach()  # STE for the optional short QAT experiment.


class FakeLinear(nn.Module):
    def __init__(self, layer, bits=4):
        super().__init__()
        self.weight = nn.Parameter(layer.weight.detach().clone())
        self.bias = nn.Parameter(layer.bias.detach().clone()) if layer.bias is not None else None
        self.bits = bits

    def forward(self, x):
        rounded = grid(self.weight, self.bits, axis=0)
        w = self.weight + (rounded - self.weight).detach()
        return F.linear(x, w, self.bias)


@torch.no_grad()
def weight_round(model, bits=4):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            layer.weight.copy_(grid(layer.weight, bits, axis=0))
    return model


class QuantAttention(nn.Module):
    """timm Attention with an explicit simulated quantiser after softmax."""
    def __init__(self, old, twin):
        super().__init__()
        self.qkv, self.proj = old.qkv, old.proj
        self.q_norm, self.k_norm = old.q_norm, old.k_norm
        self.attn_drop, self.proj_drop = old.attn_drop, old.proj_drop
        self.num_heads, self.scale = old.num_heads, old.scale
        self.prob_grid = ActivationGrid(4, twin, unsigned=True)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        prob = ((self.q_norm(q) * self.scale) @ self.k_norm(k).transpose(-2, -1)).softmax(-1)
        x = (self.attn_drop(self.prob_grid(prob)) @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


def fake_vit(model, calibration, twin=False, qat=False, steps=20, targets=None):
    # W4 + A4 at LN2 outputs, post-GELU outputs, and attention probabilities.
    # Remaining activations, patch convolution, normalisation and adds are FP32.
    for block in model.blocks:
        block.norm2 = nn.Sequential(block.norm2, ActivationGrid(4, twin))
        block.mlp.act = nn.Sequential(block.mlp.act, ActivationGrid(4, twin))
        block.attn = QuantAttention(block.attn, twin)
    with torch.inference_mode():
        for batch in batches(calibration):
            model(batch)
    for layer in model.modules():
        if isinstance(layer, ActivationGrid): layer.calibrating = False
    if not qat:
        return weight_round(model), []
    for name, layer in list(model.named_modules()):
        if isinstance(layer, nn.Linear):
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, FakeLinear(layer))
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    trace = []
    model.eval()
    for i in range(steps):
        j = (i * 4) % len(calibration)
        out = model(calibration[j:j+4])
        loss = F.cross_entropy(out, targets[j:j+4])
        optim.zero_grad(); loss.backward(); optim.step()
        trace.append({"step": i, "loss": loss.item()})
    # Freeze the simulated weights; no extra weight-rounding cost at inference.
    for name, layer in list(model.named_modules()):
        if isinstance(layer, FakeLinear):
            parent_name, _, attr = name.rpartition(".")
            plain = nn.Linear(layer.weight.shape[1], layer.weight.shape[0], bias=layer.bias is not None)
            with torch.no_grad():
                plain.weight.copy_(grid(layer.weight, layer.bits, axis=0))
                if layer.bias is not None: plain.bias.copy_(layer.bias)
            setattr(model.get_submodule(parent_name) if parent_name else model, attr, plain)
    return model.eval(), trace


@torch.no_grad()
def scale_reparameterise(model, calibration):
    """Exact LN-to-linear channel rescaling, before low-bit quantisation.

    Related core idea to RepQ-ViT; this is a simple SmoothQuant-style equalisation,
    without RepQ's full log-softmax quantiser or scale reparameterisation algorithm.
    """
    stats, handles = {}, []
    for i, block in enumerate(model.blocks):
        for key, ln in ((f"{i}.attn", block.norm1), (f"{i}.mlp", block.norm2)):
            def hook(module, args, output, key=key):
                v = output.abs().flatten(0, -2).amax(0)
                stats[key] = torch.maximum(stats.get(key, torch.zeros_like(v)), v)
            handles.append(ln.register_forward_hook(hook))
    for batch in batches(calibration): model(batch)
    for h in handles: h.remove()
    before = model(calibration[:2])
    for i, block in enumerate(model.blocks):
        for key, ln, linear in ((f"{i}.attn", block.norm1, block.attn.qkv),
                                (f"{i}.mlp", block.norm2, block.mlp.fc1)):
            a = stats[key].clamp_min(1e-5)
            w = linear.weight.abs().amax(0).clamp_min(1e-5)
            s = (a / w).sqrt().clamp(.1, 10)
            ln.weight.div_(s); ln.bias.div_(s); linear.weight.mul_(s[None])
    error = (before - model(calibration[:2])).abs().max().item()
    if error > 1e-3: raise RuntimeError(f"Scale rewrite changed FP32 outputs: {error}")
    return model, error


def reconstruct(model, calibration, steps=30, adaptive=False):
    """Reconstruct one ViT block at a time. Equal budget; uniform baseline is W4A32.

    Block mode learns each Linear output bias (simple BRECQ-inspired correction).
    Adaptive mode learns continuous rounding decisions, then hardens to adjacent
    grid points (small AdaRound-inspired objective, not the paper's full recipe).
    """
    trace = []
    with torch.no_grad():
        x = model._pos_embed(model.patch_embed(calibration[:8]))
        x = model.norm_pre(model.patch_drop(x))
    for i, original in enumerate(list(model.blocks)):
        target_block = copy.deepcopy(original).eval()
        with torch.no_grad(): target = target_block(x)
        if adaptive:
            for name, layer in list(original.named_modules()):
                if isinstance(layer, nn.Linear):
                    parent_name, _, attr = name.rpartition(".")
                    setattr(original.get_submodule(parent_name) if parent_name else original, attr, AdaptiveLinear(layer))
            params = [m.alpha for m in original.modules() if isinstance(m, AdaptiveLinear)]
        else:
            weight_round(original)
            for p in original.parameters(): p.requires_grad_(False)
            params = []
            for layer in original.modules():
                if isinstance(layer, nn.Linear) and layer.bias is not None:
                    layer.bias.requires_grad_(True); params.append(layer.bias)
        optim = torch.optim.Adam(params, lr=.01 if adaptive else .001)
        with torch.no_grad(): start = F.mse_loss(original(x), target).item()
        for step in range(steps):
            loss = F.mse_loss(original(x), target)
            if adaptive:
                # Gradually encourage decisions to reach 0 or 1.
                reg = sum((m.alpha.sigmoid() * (1-m.alpha.sigmoid())).mean() for m in original.modules() if isinstance(m, AdaptiveLinear))
                loss = loss + .001 * (step / max(1, steps-1)) * reg
            optim.zero_grad(); loss.backward(); optim.step()
        if adaptive:
            for name, layer in list(original.named_modules()):
                if isinstance(layer, AdaptiveLinear):
                    parent_name, _, attr = name.rpartition(".")
                    plain = nn.Linear(layer.floor.shape[1], layer.floor.shape[0], bias=layer.bias is not None)
                    with torch.no_grad():
                        plain.weight.copy_((layer.floor + (layer.alpha >= 0).float()) * layer.scale)
                        if layer.bias is not None: plain.bias.copy_(layer.bias)
                    setattr(original.get_submodule(parent_name) if parent_name else original, attr, plain)
        with torch.no_grad():
            end = F.mse_loss(original(x), target).item()
            x = original(x).detach()
        trace.append({"block": i, "mse_before": start, "mse_after": end, "steps": steps})
    with torch.no_grad(): model.head.weight.copy_(grid(model.head.weight, 4, axis=0))
    return model.eval(), trace


class AdaptiveLinear(nn.Module):
    def __init__(self, layer):
        super().__init__()
        w = layer.weight.detach()
        self.register_buffer("scale", w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 7)
        self.register_buffer("floor", torch.floor(w / self.scale))
        fraction = (w / self.scale - self.floor).clamp(.001, .999)
        self.alpha = nn.Parameter(torch.logit(fraction))
        self.bias = nn.Parameter(layer.bias.detach().clone(), requires_grad=False) if layer.bias is not None else None

    def forward(self, x):
        return F.linear(x, (self.floor + self.alpha.sigmoid()) * self.scale, self.bias)

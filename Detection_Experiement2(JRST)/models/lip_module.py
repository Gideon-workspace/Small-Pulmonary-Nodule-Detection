"""
LIP (Local Importance-based Pooling, Gao et al., ICCV 2019) swaps for a
torchvision ResNet, written to reproduce the official LIP repo's ResNet:

  * same math      : lip2d, 12*sigmoid gate, InstanceNorm logit modules
  * same wiring    : stride-2 Bottleneck = shared trunk (1x1 -> 3x3, width 128,
                     computed ONCE from the block input) + per-branch 1x1 head;
                     residual: conv1 -> LIP -> 1x1 -> bn2; shortcut: LIP -> 1x1 -> bn
  * same names     : bottleneck_shared / conv2.0.postprocessing / downsample.lip ...
                     so the repo's state_dict keys line up (see load_repo_checkpoint)
  * same init      : kaiming(fan_out) convs, zero-init LIP heads, zero bn3,
                     fc ~ N(0, 0.01)   (repo_init=True)

Everything is still controlled per-piece by apply_lip():

    stem_conv : conv1 (7x7, s2) -> LIP + 7x7 conv, s1       (NOT in the repo)
    stem_pool : MaxPool         -> LIPModule                (repo: SimplifiedLIP)
    layers    : first block of layer2/3/4, any subset       (repo: all three)

Exact repo configuration:
    model = resnet50(weights=None)
    apply_lip(model, stem_conv=False, stem_pool=True, layers=(2, 3, 4),
              repo_init=True)

Call apply_lip BEFORE model.to(device) and before building the optimizer.
Supports torchvision Bottleneck (ResNet-50/101/152) as in the repo, plus
BasicBlock (ResNet-18/34) as an extension. ResNeXt / Wide-ResNet: not supported.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

BOTTLENECK_WIDTH = 128
COEFF = 12.0


# =============================================================================
# 1. Repo primitives (kept identical to the official code)
# =============================================================================

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return (F.avg_pool2d(x * weight, kernel, stride, padding) /
            F.avg_pool2d(weight, kernel, stride, padding))


class SoftGate(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x).mul(COEFF)


class BottleneckShared(nn.Module):
    """Shared logit trunk. Runs once per stride-2 block, on the block input."""

    def __init__(self, channels, width=BOTTLENECK_WIDTH):
        super().__init__()
        self.logit = nn.Sequential(OrderedDict((
            ('conv1', conv1x1(channels, width)),
            ('bn1', nn.InstanceNorm2d(width, affine=True)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', conv3x3(width, width)),
            ('bn2', nn.InstanceNorm2d(width, affine=True)),
            ('relu2', nn.ReLU(inplace=True)),
        )))
        for m in self.modules():                      # repo: kaiming on all convs
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')

    def forward(self, x):
        return self.logit(x)


class BottleneckLIP(nn.Module):
    """Branch-specific head: trunk features -> logits (channels) -> LIP."""

    def __init__(self, channels, width=BOTTLENECK_WIDTH):
        super().__init__()
        self.postprocessing = nn.Sequential(OrderedDict((
            ('conv', conv1x1(width, channels)),
            ('bn', nn.InstanceNorm2d(channels, affine=True)),
            ('gate', SoftGate()),
        )))
        self.init_layer()

    def init_layer(self):
        self.postprocessing[0].weight.data.fill_(0.0)   # starts as avg-pooling

    def forward_with_shared(self, x, shared):
        return lip2d(x, self.postprocessing(shared))


# =============================================================================
# 2. Stem pieces (LIPModule is your class, unchanged)
# =============================================================================

class LIPModule(nn.Module):
    """
    Local Importance-based Pooling (LIP)
    Based on Gao et al., ICCV 2019.

    Used here to replace the ResNet stem MaxPool:
        kernel_size = 3
        stride = 2
        padding = 1
    """

    def __init__(self, channels, kernel_size=3, stride=2, padding=1):
        super().__init__()

        # For the MaxPool replacement, the paper uses
        # a single 3x3 convolution as the logit module.
        self.logit = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Affine Instance Normalization
        self.norm = nn.InstanceNorm2d(
            channels,
            affine=True
        )

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize LIP to behave like average pooling.
        nn.init.zeros_(self.logit.weight)

    def forward(self, x):
        # 1. Learn the local importance logits
        logit = self.logit(x)

        # 2. Instance normalization
        logit = self.norm(logit)

        # 3. Amplified sigmoid
        logit = 12.0 * torch.sigmoid(logit)

        # 4. Convert logits into positive importance weights
        weight = torch.exp(logit)

        # 5. Local importance-weighted aggregation
        numerator = F.avg_pool2d(
            x * weight,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        denominator = F.avg_pool2d(
            weight,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        # 6. Normalize locally
        output = numerator / (denominator + 1e-8)

        return output


class LIPStemConv(nn.Module):
    """Replaces conv1 (7x7, stride 2): LIP downsample -> 7x7 conv, stride 1."""

    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.lip = LIPModule(in_ch)                       # your class, unchanged
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=7,
                              stride=1, padding=3, bias=False)

    def forward(self, x):
        return self.conv(self.lip(x))


# =============================================================================
# 3. Stride-2 blocks (repo structure, wrapped around torchvision blocks)
# =============================================================================

def _post_conv(cin, cout, k, old_conv=None):
    """Stride-1 conv after LIP. Copies old weights if shapes match, else kaiming."""
    conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
    if old_conv is not None and old_conv.weight.shape == conv.weight.shape:
        conv.weight.data.copy_(old_conv.weight.data)
    else:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
    return conv


def _lip_downsample(inplanes, width, old_downsample):
    """Repo shortcut: LIP (on inplanes ch) -> 1x1 conv (stride 1) -> BN."""
    old_conv, old_bn = old_downsample[0], old_downsample[1]
    return nn.Sequential(OrderedDict((
        ('lip', BottleneckLIP(inplanes, width)),
        ('conv', _post_conv(inplanes, old_conv.out_channels, 1, old_conv)),
        ('bn', old_bn),
    )))


class LIPDownBottleneck(nn.Module):
    """Stride-2 Bottleneck, identical in structure/forward to the repo's."""

    def __init__(self, blk, width=BOTTLENECK_WIDTH, post_k=1):
        super().__init__()
        inplanes = blk.conv1.in_channels
        mid = blk.conv2.out_channels

        self.bottleneck_shared = BottleneckShared(inplanes, width)

        self.conv1, self.bn1 = blk.conv1, blk.bn1
        self.conv2 = nn.Sequential(
            BottleneckLIP(mid, width),
            _post_conv(mid, mid, post_k, blk.conv2),   # repo: 1x1
        )
        self.bn2 = blk.bn2
        self.conv3, self.bn3 = blk.conv3, blk.bn3
        self.relu = nn.ReLU(inplace=True)

        self.downsample = _lip_downsample(inplanes, width, blk.downsample)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))

        shared = self.bottleneck_shared(x)                 # computed once
        out = self.conv2[0].forward_with_shared(out, shared)
        out = self.conv2[1](out)

        out = self.relu(self.bn2(out))
        out = self.bn3(self.conv3(out))

        residual = self.downsample[0].forward_with_shared(x, shared)
        residual = self.downsample[1](residual)
        residual = self.downsample[2](residual)

        out += residual
        return self.relu(out)


class LIPDownBasicBlock(nn.Module):
    """
    Stride-2 BasicBlock (ResNet-18/34). The repo has no working BasicBlock, so this
    is an extension built the same way: LIP on the block input replaces the strided
    3x3 conv1 (LIP -> 3x3 stride 1), shortcut as in the Bottleneck.
    """

    def __init__(self, blk, width=BOTTLENECK_WIDTH, post_k=3):
        super().__init__()
        inplanes = blk.conv1.in_channels
        planes = blk.conv1.out_channels

        self.bottleneck_shared = BottleneckShared(inplanes, width)

        self.conv1 = nn.Sequential(
            BottleneckLIP(inplanes, width),
            _post_conv(inplanes, planes, post_k, blk.conv1),
        )
        self.bn1 = blk.bn1
        self.relu = nn.ReLU(inplace=True)
        self.conv2, self.bn2 = blk.conv2, blk.bn2

        self.downsample = _lip_downsample(inplanes, width, blk.downsample)

    def forward(self, x):
        shared = self.bottleneck_shared(x)

        out = self.conv1[0].forward_with_shared(x, shared)
        out = self.conv1[1](out)
        out = self.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))

        residual = self.downsample[0].forward_with_shared(x, shared)
        residual = self.downsample[1](residual)
        residual = self.downsample[2](residual)

        out += residual
        return self.relu(out)


# =============================================================================
# 4. Repo initialisation (mirrors the repo's ResNet.__init__ exactly)
# =============================================================================

def _reinit_like_repo(model):
    # 1) kaiming on every conv, BN to (1, 0)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # 2) classifier
    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.zero_()

    # 3) init_layer() hooks: zero last BN gamma of each block, zero LIP heads
    for m in model.modules():
        if isinstance(m, (Bottleneck, LIPDownBottleneck)):
            m.bn3.weight.data.zero_()
        elif isinstance(m, (BasicBlock, LIPDownBasicBlock)):
            m.bn2.weight.data.zero_()          # repo's (commented) BasicBlock did this
        elif isinstance(m, BottleneckLIP):
            m.init_layer()
        elif isinstance(m, LIPModule):
            nn.init.zeros_(m.logit.weight)


# =============================================================================
# 5. The switchboard
# =============================================================================

def apply_lip(model, stem_conv=False, stem_pool=False, layers=(),
              width=BOTTLENECK_WIDTH, post_k=None, repo_init=False):
    """
    Swap downsampling steps of a torchvision ResNet for LIP, in place.

    stem_conv : conv1 -> LIPStemConv (extension, not in the repo)
    stem_pool : MaxPool -> LIPModule (repo: SimplifiedLIP)
    layers    : e.g. (2, 3, 4) or ("layer2",); swaps the first block of each
    width     : shared-trunk width (repo: 128)
    post_k    : conv kernel after LIP in the residual branch.
                Default 1 for Bottleneck (repo), 3 for BasicBlock.
    repo_init : re-initialise the WHOLE model exactly like the repo (kaiming convs,
                zero LIP heads, zero bn3, fc ~ N(0, 0.01)). For training from
                scratch only; it overwrites any pretrained weights.
    """
    if stem_conv:
        old = model.conv1
        if not (isinstance(old, nn.Conv2d) and old.stride == (2, 2)):
            raise ValueError("conv1 is not the original stride-2 conv "
                             "(already swapped?)")
        new = LIPStemConv(old.in_channels, old.out_channels)
        new.conv.weight.data.copy_(old.weight.data)
        model.conv1 = new

    if stem_pool:
        if isinstance(model.maxpool, LIPModule):
            raise ValueError("maxpool is already a LIPModule")
        model.maxpool = LIPModule(model.bn1.num_features)

    for l in layers:
        name = l if isinstance(l, str) else f"layer{l}"
        blk = getattr(model, name)[0]

        if isinstance(blk, Bottleneck):
            if blk.conv2.stride != (2, 2):
                raise ValueError(f"{name} does not downsample "
                                 f"(layer1 never does); nothing to swap")
            if blk.conv2.groups != 1 or blk.conv2.dilation != (1, 1):
                raise ValueError("grouped/dilated Bottleneck not supported")
            new_blk = LIPDownBottleneck(
                blk, width, post_k if post_k is not None else 1)

        elif isinstance(blk, BasicBlock):
            if blk.conv1.stride != (2, 2):
                raise ValueError(f"{name} does not downsample "
                                 f"(layer1 never does); nothing to swap")
            new_blk = LIPDownBasicBlock(
                blk, width, post_k if post_k is not None else 3)

        else:
            raise ValueError(f"{name}[0] is {type(blk).__name__}; "
                             f"already swapped or unsupported")

        getattr(model, name)[0] = new_blk

    if repo_init:
        _reinit_like_repo(model)

    return model


# =============================================================================
# 6. Loading the repo's checkpoint (repo config only: stem_pool + layers 2,3,4)
# =============================================================================

def load_repo_checkpoint(model, path):
    """
    Loads the official repo's state_dict into a model built with
        apply_lip(resnet50(weights=None), stem_pool=True, layers=(2, 3, 4))
    Block keys already match; only the stem MaxPool and layer1's plain
    shortcut are named differently and get remapped. strict=True, so any
    structural mismatch is reported instead of silently skipped.
    """
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    own = model.state_dict()

    renames = ((".downsample.conv.", ".downsample.0."),
               (".downsample.bn.", ".downsample.1."),
               ("maxpool.logit.conv.", "maxpool.logit."),
               ("maxpool.logit.bn.", "maxpool.norm."))

    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k not in own:
            for a, b in renames:
                k2 = k.replace(a, b)
                if k2 in own:
                    k = k2
                    break
        out[k] = v
    return model.load_state_dict(out, strict=True)


# =============================================================================
# 7. Self-checks:  python lip_resnet.py
# =============================================================================

if __name__ == "__main__":
    from torchvision.models import resnet18, resnet50

    # (a) zero logits => LIP is plain average pooling over the valid window
    x = torch.randn(2, 8, 16, 16)
    assert torch.allclose(lip2d(x, torch.zeros_like(x)),
                          F.avg_pool2d(x, 3, 2, 1, count_include_pad=False),
                          atol=1e-5)

    # (b) repo layout + repo init on ResNet-50
    m = apply_lip(resnet50(weights=None), stem_pool=True, layers=(2, 3, 4),
                  repo_init=True)
    keys = set(m.state_dict().keys())
    for k in ("layer2.0.bottleneck_shared.logit.conv1.weight",
              "layer2.0.bottleneck_shared.logit.conv2.weight",
              "layer2.0.conv2.0.postprocessing.conv.weight",
              "layer2.0.conv2.0.postprocessing.bn.weight",
              "layer2.0.conv2.1.weight",
              "layer2.0.downsample.lip.postprocessing.conv.weight",
              "layer2.0.downsample.conv.weight",
              "layer2.0.downsample.bn.weight"):
        assert k in keys, k
    assert m.layer2[0].bn3.weight.abs().sum() == 0
    assert m.layer3[0].conv2[0].postprocessing[0].weight.abs().sum() == 0
    assert m.maxpool.logit.weight.abs().sum() == 0
    print("repo layout + init: OK")

    # (c) shapes for several swap combinations
    configs = {
        "everything":    dict(stem_conv=True,  stem_pool=True,  layers=(2, 3, 4)),
        "pool + layer2": dict(stem_conv=False, stem_pool=True,  layers=(2,)),
        "layer3 only":   dict(stem_conv=False, stem_pool=False, layers=(3,)),
        "conv1 only":    dict(stem_conv=True,  stem_pool=False, layers=()),
    }
    for ctor in (resnet18, resnet50):
        for label, cfg in configs.items():
            m = apply_lip(ctor(weights=None), **cfg).eval()
            with torch.no_grad():
                x = m.relu(m.bn1(m.conv1(torch.randn(2, 3, 224, 224))))
                s = [tuple(x.shape[1:])]
                x = m.maxpool(x)
                s.append(tuple(x.shape[1:]))
                for n in ("layer1", "layer2", "layer3", "layer4"):
                    x = getattr(m, n)(x)
                    s.append(tuple(x.shape[1:]))
                out = m(torch.randn(2, 3, 224, 224))
            print(f"{ctor.__name__:9s} {label:14s} {s}  logits={tuple(out.shape)}")

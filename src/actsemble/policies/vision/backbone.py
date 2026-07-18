"""Vision backbones for the image (high-dim) policies — CODE-READY, NOT TRAINED.

A self-contained ResNet-18 (torchvision-faithful BasicBlock, layers [2,2,2,2]) +
spatial-softmax keypoint pooling — the standard Diffusion-Policy / ACT vision
setup. Self-contained (no torchvision) because torchvision's compiled ops are ABI-
incompatible with this torch+cu128 build; a torch-compatible torchvision ResNet18
can be swapped in at benchmark time (see docs/deferred_work.md).

Two consumers, gated by ``observation.mode: rgb``:
* ``ImageObsEncoder`` -> a per-frame feature vector for the Diffusion / Flow
  U-Net ``global_cond`` (spatial-softmax keypoints ++ optional proprio state);
* ``image_feature_map`` -> a ``[B, h*w, C]`` token sequence for the ACT decoder
  memory (with 2D sinusoidal position embeddings, added at integration time).

The env rgb obs mode + image dataset + training loop are the DEFERRED data
pipeline; only the architecture lives here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(i, o, stride=1):
    return nn.Conv2d(i, o, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + identity, inplace=True)


class ResNet18(nn.Module):
    """torchvision-faithful ResNet-18 trunk (no avgpool/fc): image -> feature map."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._layer(64, 64, 2, 1)
        self.layer2 = self._layer(64, 128, 2, 2)
        self.layer3 = self._layer(128, 256, 2, 2)
        self.layer4 = self._layer(256, 512, 2, 2)
        self.out_channels = 512
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _layer(in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
        layers += [BasicBlock(out_ch, out_ch) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.maxpool(F.relu(self.bn1(self.conv1(x)), inplace=True))
        return self.layer4(self.layer3(self.layer2(self.layer1(h))))  # [B, 512, h, w]


class SpatialSoftmax(nn.Module):
    """Diffusion-Policy spatial-softmax: feature map -> ``num_kp`` (x, y) keypoints."""

    def __init__(self, channels: int, num_kp: int = 32):
        super().__init__()
        self.proj = (
            nn.Conv2d(channels, num_kp, 1) if num_kp != channels else nn.Identity()
        )
        self.num_kp = num_kp

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        z = self.proj(feat).reshape(b, self.num_kp, h * w)
        attn = F.softmax(z, dim=-1)
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, h, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1, 1, w, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        grid = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=0)  # [2, h*w]
        kp = torch.einsum("bkp,dp->bkd", attn, grid)  # [B, num_kp, 2]
        return kp.reshape(b, self.num_kp * 2)


class ImageObsEncoder(nn.Module):
    """ResNet18 + spatial-softmax -> per-frame feature vector, concatenated over
    the observation horizon and (optionally) with proprioceptive state. Produces
    the ``global_cond`` for the Diffusion / Flow U-Net (image policies)."""

    def __init__(self, *, in_channels: int = 3, num_kp: int = 32, proprio_dim: int = 0):
        super().__init__()
        self.backbone = ResNet18(in_channels)
        self.pool = SpatialSoftmax(self.backbone.out_channels, num_kp)
        self.feature_dim = num_kp * 2 + int(proprio_dim)
        self.proprio_dim = int(proprio_dim)

    def forward(
        self, images: torch.Tensor, proprio: torch.Tensor | None = None
    ) -> torch.Tensor:
        """images [B, H_o, C, H, W] (+ proprio [B, H_o, proprio_dim]) -> [B, H_o*feature_dim]."""
        b, ho = images.shape[:2]
        feat = self.pool(self.backbone(images.flatten(0, 1))).reshape(
            b, ho, -1
        )  # [B,H_o,num_kp*2]
        if self.proprio_dim and proprio is None:
            raise ValueError(f"proprio is required when proprio_dim={self.proprio_dim}")
        if proprio is not None:
            expected = (b, ho, self.proprio_dim)
            if tuple(proprio.shape) != expected:
                raise ValueError(
                    f"proprio shape {tuple(proprio.shape)} != expected {expected}"
                )
            feat = torch.cat([feat, proprio], dim=-1)
        return feat.reshape(b, -1)


def image_feature_map(backbone: ResNet18, images: torch.Tensor) -> torch.Tensor:
    """images [B, C, H, W] -> [B, h*w, C] token sequence for the ACT decoder memory."""
    fmap = backbone(images)  # [B, C, h, w]
    b, c, h, w = fmap.shape
    return fmap.reshape(b, c, h * w).transpose(1, 2)  # [B, h*w, C]

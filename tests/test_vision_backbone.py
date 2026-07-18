"""Shape tests for the image-policy vision backbone (code-ready, not trained)."""

from __future__ import annotations

import torch

from actsemble.policies.vision.backbone import (
    ImageObsEncoder,
    ResNet18,
    SpatialSoftmax,
    image_feature_map,
)


def test_resnet18_feature_map_shape():
    m = ResNet18().eval()
    f = m(torch.randn(2, 3, 96, 96))  # 96 -> /2 conv -> /2 pool -> /2/2/2 layers = 3x3
    assert f.shape == (2, 512, 3, 3)


def test_spatial_softmax_keypoints():
    kp = SpatialSoftmax(512, num_kp=16)(torch.randn(2, 512, 6, 6))
    assert kp.shape == (2, 32)  # 16 keypoints x (x, y)


def test_image_obs_encoder_global_cond():
    enc = ImageObsEncoder(num_kp=16, proprio_dim=4).eval()
    out = enc(torch.randn(2, 2, 3, 96, 96), torch.randn(2, 2, 4))  # [B, H_o, C, H, W]
    assert enc.feature_dim == 16 * 2 + 4
    assert out.shape == (2, 2 * enc.feature_dim)  # concatenated over the obs horizon


def test_image_feature_map_tokens_for_act():
    toks = image_feature_map(ResNet18().eval(), torch.randn(2, 3, 96, 96))
    assert toks.shape == (2, 9, 512)  # [B, h*w, C] token sequence

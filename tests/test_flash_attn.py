import pytest
import numpy as np
from torchstudio.attention.flash_attn import (
    softmax,
    full_attention,
    flash_attention,
)

@pytest.mark.parametrize('shape', [(32, 16), (64, 32), (127, 53)])
def test_full_attention(shape):
    q = np.random.randn(*shape)
    k = np.random.randn(*shape)
    v = np.random.randn(*shape)
    score = full_attention(q, k, v)
    gt = softmax(q @ k.T, axis=1) @ v
    assert score.shape == gt.shape
    assert (np.abs(gt - score) < 1e-6).all()

@pytest.mark.parametrize(
    'shape, M', [
        ((32, 16), 256), ((64, 32), 233), ((127, 53), 677),
        ((32, 64), 128), ((64, 128), 1024), ((127, 256), 512),
    ]
)
def test_flash_attention(shape, M):
    q = np.random.randn(*shape)
    k = np.random.randn(*shape)
    v = np.random.randn(*shape)
    score = flash_attention(q, k, v, M)
    gt = softmax(q @ k.T, axis=1) @ v
    assert score.shape == gt.shape
    assert (np.abs(gt - score) < 1e-6).all()
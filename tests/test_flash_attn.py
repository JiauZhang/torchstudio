import pytest
import numpy as np
from torchstudio.attention.flash_attn import (
    softmax, tiled_softmax,
    full_attention,
    flash_attention,
    flash_attention_v2,
)

@pytest.mark.parametrize('shape', [(1, 6), (32, 33), (64, 87), (68, 53)])
def test_tiled_softmax(shape):
    x = np.random.randn(*shape)
    sep = shape[1] // 2 + 2
    y = x[:, :sep]
    z = x[:, sep:]
    o = tiled_softmax(y, z, 1)
    gt = softmax(x, 1)
    assert o.shape == gt.shape
    assert (np.abs(np.sum(o, axis=1) - 1) < 1e-6).all()
    assert (np.abs(gt - o) < 1e-6).all()

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

    score_v2 = flash_attention_v2(q, k, v, M)
    assert score_v2.shape == gt.shape
    assert (np.abs(gt - score_v2) < 1e-6).all()
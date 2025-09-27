import pytest
import numpy as np
from torchstudio.utils.iou import batch_iou

@pytest.mark.parametrize(
    'boxes1, boxes2', [
        (
            np.array([
                [1, 2, 7, 8],
                [3, 4, 9, 9],
            ]), np.array([
                [1, 2, 7, 8],
                [3, 4, 9, 9],
                [100, 110, 200, 233],
            ]),
        ),
    ]
)
def test_iou(boxes1, boxes2):
    iou = batch_iou(boxes1, boxes2, epsilon=0.0)
    iou_gt = np.array([
        [1, 16/50, 0],
        [16/50, 1, 0],
    ])
    assert iou.shape == iou_gt.shape
    assert (np.abs(iou - iou_gt) < 1e-9).all()

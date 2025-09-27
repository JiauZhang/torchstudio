import numpy as np

def batch_iou(boxes1, boxes2, epsilon=1e-6):
    boxes1 = np.expand_dims(boxes1, 1) # (N, 1, 4)
    boxes2 = np.expand_dims(boxes2, 0) # (1, M, 4)

    x_left = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y_top = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x_right = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y_bottom = np.minimum(boxes1[..., 3], boxes2[..., 3])
    intersection = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union = area1 + area2 - intersection
    iou = intersection / (union + epsilon)

    return iou

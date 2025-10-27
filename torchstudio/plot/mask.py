import cv2
import numpy as np

def draw_mask_edge(image: np.ndarray, mask: np.ndarray, color=(208, 160, 72)):
    contours, _ = cv2.findContours(
        mask,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(image, contours, -1, color=color, thickness=2)

def draw_mask(image: np.ndarray, mask: np.ndarray, color=(0, 128, 0), alpha=0.4):
    image[mask > 0] = (1 - alpha) * image[mask > 0] + np.array(color)

if __name__ == '__main__':
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[128:256, 128:256] = 255
    draw_mask(image, mask)
    draw_mask_edge(image, mask)
    cv2.imwrite('mask-edge.png', image)

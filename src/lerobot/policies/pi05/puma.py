from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# Keep sidecar targets in the observation namespace so LeRobot's
# batch-to-transition preprocessor preserves and moves them to the policy device.
PUMA_FUTURE_FEATURES = "observation.puma.future_features"
PUMA_FUTURE_VALID = "observation.puma.future_valid"


def dense_flow_rgb(
    historical: Tensor,
    current: Tensor,
    resolution: tuple[int, int] = (64, 64),
    magnitude_percentile: float = 99.0,
    noise_threshold: float = 0.05,
) -> Tensor:
    """Compute PUMA's Farneback RGB flow maps for batched images in [0, 1]."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("PUMA optical flow requires opencv-python-headless") from exc

    if historical.ndim != 5 or current.ndim != 4:
        raise ValueError("Expected historical BCHW sequence and current BCHW image")
    if historical.shape[0] != current.shape[0]:
        raise ValueError("Historical and current image batch sizes differ")

    historical_np = historical.detach().float().cpu().numpy()
    current_np = current.detach().float().cpu().numpy()
    output = np.zeros((historical.shape[0], historical.shape[1], 3, *resolution), dtype=np.float32)

    def gray(image: np.ndarray) -> np.ndarray:
        image = np.moveaxis(image, 0, -1)
        image = cv2.resize(image, (resolution[1], resolution[0]), interpolation=cv2.INTER_AREA)
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    for batch_idx in range(historical.shape[0]):
        current_gray = gray(current_np[batch_idx])
        for history_idx in range(historical.shape[1]):
            old_gray = gray(historical_np[batch_idx, history_idx])
            flow = cv2.calcOpticalFlowFarneback(
                old_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
            scale = float(np.percentile(magnitude, magnitude_percentile))
            normalized = np.clip(magnitude / max(scale, 1e-6), 0.0, 1.0)
            normalized[normalized < noise_threshold] = 0.0
            hsv = np.zeros((*resolution, 3), dtype=np.uint8)
            hsv[..., 0] = np.mod(angle / 2.0, 180).astype(np.uint8)
            hsv[..., 1] = 255
            hsv[..., 2] = np.round(normalized * 255.0).astype(np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            output[batch_idx, history_idx] = np.moveaxis(rgb, -1, 0)

    return torch.from_numpy(output).to(device=current.device, dtype=current.dtype)

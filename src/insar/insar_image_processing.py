from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import (
    INSAR_IMAGE_MASK_THRESHOLD_QUANTILE,
    INSAR_MASK_DIR,
    INSAR_MASK_METADATA_PATH,
    INSAR_OVERLAY_DIR,
    INSAR_RAW_IMAGE_DIR,
)


def _load_grayscale(path):
    img = mpimg.imread(path)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return img.astype(float)


def _normalized(arr):
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    return (arr - arr_min) / (arr_max - arr_min + 1e-9)


def process_insar_images(
    image_dir=INSAR_RAW_IMAGE_DIR,
    metadata_output=INSAR_MASK_METADATA_PATH,
    mask_dir=INSAR_MASK_DIR,
    overlay_dir=INSAR_OVERLAY_DIR,
):
    image_dir = Path(image_dir)
    files = sorted(
        [
            p
            for p in image_dir.glob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        ]
    )

    if not files:
        raise FileNotFoundError(f"No InSAR images found in {image_dir}")

    mask_dir = Path(mask_dir)
    overlay_dir = Path(overlay_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_grayscale(files[0])
    baseline = _normalized(baseline)

    rows = []
    for image_path in files:
        current = _normalized(_load_grayscale(image_path))
        deformation = np.abs(current - baseline)

        positive = deformation[deformation > 0]
        if positive.size == 0:
            mask = np.zeros_like(deformation, dtype=np.uint8)
        else:
            threshold = np.quantile(positive, INSAR_IMAGE_MASK_THRESHOLD_QUANTILE)
            mask = (deformation >= threshold).astype(np.uint8)

        mask_path = mask_dir / f"mask_{image_path.stem}.png"
        overlay_path = overlay_dir / f"overlay_{image_path.stem}.png"

        plt.imsave(mask_path, mask, cmap="gray")

        overlay = np.zeros((current.shape[0], current.shape[1], 3), dtype=float)
        overlay[..., 0] = current
        overlay[..., 1] = current
        overlay[..., 2] = current
        overlay[mask == 1] = [1.0, 0.1, 0.1]
        plt.imsave(overlay_path, overlay)

        parts = image_path.stem.split("_")
        ts = pd.NaT
        if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 8:
            ts = pd.to_datetime(parts[-1], format="%Y%m%d", errors="coerce")

        rows.append(
            {
                "timestamp": ts,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "mask_ratio": float(mask.mean()),
            }
        )

    metadata = pd.DataFrame(rows).sort_values("timestamp", na_position="last")
    metadata_output = Path(metadata_output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_output, index=False)
    return metadata


if __name__ == "__main__":
    process_insar_images()
    print("InSAR image mask prediction completed")

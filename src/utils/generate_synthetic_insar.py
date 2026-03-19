from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import INSAR_PROCESSED_PATH, INSAR_RAW_IMAGE_DIR, RANDOM_SEED


def _build_deformation_field(size, center_shift, intensity):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    cx = 0.1 * np.sin(center_shift)
    cy = 0.1 * np.cos(center_shift)
    sigma = 0.23
    gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))

    return intensity * gaussian


def generate_synthetic_insar(
    output_path=INSAR_PROCESSED_PATH,
    image_dir=INSAR_RAW_IMAGE_DIR,
    periods=30,
    image_size=128,
    seed=RANDOM_SEED,
):
    rng = np.random.default_rng(seed + 1)
    dates = pd.date_range("2024-01-01", periods=periods, freq="12D")

    los = np.cumsum(rng.normal(0.2, 0.3, len(dates)))
    damage_start = int(periods * 0.5)
    los[damage_start:] += np.linspace(0, 6, len(los[damage_start:]))

    ts_df = pd.DataFrame(
        {
            "timestamp": dates,
            "los_displacement": los,
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ts_df.to_csv(output_path, index=False)

    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    for ts, los_value in zip(dates, los):
        background = rng.normal(0.0, 0.03, (image_size, image_size))
        intensity = max(0.0, los_value / 8.0)
        deformation = _build_deformation_field(
            size=image_size,
            center_shift=(ts.toordinal() % 365) / 365.0 * 2 * np.pi,
            intensity=intensity,
        )
        image = background + deformation

        # Normalize to [0, 1] for stable PNG output.
        image = (image - image.min()) / (image.max() - image.min() + 1e-9)
        file_name = f"sar_{ts.strftime('%Y%m%d')}.png"
        plt.imsave(image_dir / file_name, image, cmap="gray")

    return ts_df


if __name__ == "__main__":
    generate_synthetic_insar()
    print("Synthetic InSAR timeseries and image frames generated")

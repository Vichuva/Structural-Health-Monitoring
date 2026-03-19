import numpy as np
import pandas as pd

from src.utils.config import GNSS_RAW_PATH, RANDOM_SEED


def generate_synthetic_gnss(output_path=GNSS_RAW_PATH, periods=60, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")

    x = 1000 + np.cumsum(rng.normal(0, 0.002, len(dates)))
    y = 2000 + np.cumsum(rng.normal(0, 0.002, len(dates)))
    z = 50 + np.cumsum(rng.normal(0, 0.001, len(dates)))

    damage_start = int(periods * 0.67)
    z[damage_start:] -= np.linspace(0, 0.010, len(z[damage_start:]))

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "x": x,
            "y": y,
            "z": z,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    generate_synthetic_gnss()
    print("Synthetic GNSS data generated")

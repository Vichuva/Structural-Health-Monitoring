from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import RANDOM_SEED, SENSOR_RAW_PATH


SENSOR_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "vibration_rms",
    "strain_ue",
    "tilt_x_deg",
    "tilt_y_deg",
    "acoustic_db",
]


def generate_synthetic_sensor_data(output_path=SENSOR_RAW_PATH, periods=120, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed + 3)
    dates = pd.date_range("2024-01-01", periods=periods, freq="12h")

    temperature = 28 + 4 * np.sin(np.linspace(0, 8 * np.pi, periods)) + rng.normal(0, 0.6, periods)
    humidity = 58 + 9 * np.sin(np.linspace(0, 4 * np.pi, periods) + 0.7) + rng.normal(0, 1.2, periods)
    vibration = 0.08 + np.abs(rng.normal(0, 0.015, periods))
    strain = np.cumsum(rng.normal(0.0, 0.4, periods))
    tilt_x = np.cumsum(rng.normal(0.0, 0.005, periods))
    tilt_y = np.cumsum(rng.normal(0.0, 0.004, periods))
    acoustic = 37 + rng.normal(0, 1.2, periods)

    anomaly_start = int(periods * 0.70)
    ramp = np.linspace(0, 1, periods - anomaly_start)
    vibration[anomaly_start:] += 0.12 * ramp
    strain[anomaly_start:] += 24 * ramp
    tilt_x[anomaly_start:] += 0.8 * ramp
    tilt_y[anomaly_start:] -= 0.6 * ramp
    acoustic[anomaly_start:] += 7 * ramp

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "temperature_c": temperature,
            "humidity_pct": np.clip(humidity, 20, 95),
            "vibration_rms": vibration,
            "strain_ue": strain,
            "tilt_x_deg": tilt_x,
            "tilt_y_deg": tilt_y,
            "acoustic_db": acoustic,
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    generate_synthetic_sensor_data()
    print("Synthetic sensor data generated")

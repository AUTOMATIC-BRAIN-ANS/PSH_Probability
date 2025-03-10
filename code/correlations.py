import itertools
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_PATH = Path("")


def setup_directories(base_path, model_name):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = Path(base_path) / f"{model_name}_{date_str}"
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def calculate_zero_crossing_rate(data):
    return ((data[:-1] * data[1:]) < 0).sum()


def calculate_standard_deviation(data):
    return np.std(data)


def calculate_mean_absolute_deviation(data):
    return np.mean(np.abs(data - np.mean(data)))


def calculate_range(data):
    return np.ptp(data)


def calculate_percentage_of_minus_sign(data):
    return np.mean(data < 0) * 100


def interpolate_missing_values(signal, threshold):
    signal = signal.copy()
    if signal.isnull().sum() == 0:
        return signal
    if signal.isnull().sum() / len(signal) > threshold:
        return None
    signal.interpolate(method="linear", limit_direction="both", inplace=True)
    return signal


def calculate_rolling_correlation(signal1, signal2, window_size_in_minutes):
    correlation = signal1.rolling(window=window_size_in_minutes, min_periods=1).corr(
        signal2
    )
    return correlation


def calculate_rolling_relationship(
    metadata_df,
    base_path,
    signal1,
    signal2,
    window_days,
    window_hours,
    interpolation_threshold=10,
):
    window_size_in_minutes = window_hours * 60
    max_period_minutes = window_days * 24 * 60

    zcr_data = metadata_df.copy()
    std_data = metadata_df.copy()
    mad_data = metadata_df.copy()
    range_data = metadata_df.copy()
    mean_data = metadata_df.copy()
    minus_sign_data = metadata_df.copy()

    for index, row in metadata_df.iterrows():
        patient_id = row["ID"]
        patient_signals = read_signals(
            patient_id, base_path
        )  # utility for reading signals

        if patient_signals.empty:
            print(f"No signals available for patient {patient_id}, skipping...")
            continue

        if (
            signal1 not in patient_signals.columns
            or signal2 not in patient_signals.columns
        ):
            print(f"Required signals missing for patient {patient_id}, skipping...")
            continue

        patient_signals[signal1] = interpolate_missing_values(
            patient_signals[signal1], interpolation_threshold
        )
        patient_signals[signal2] = interpolate_missing_values(
            patient_signals[signal2], interpolation_threshold
        )

        if patient_signals[signal1] is None or patient_signals[signal2] is None:
            print(
                f"Too many missing values to interpolate for patient {patient_id}, skipping..."
            )
            continue

        patient_signals.dropna(subset=[signal1, signal2], inplace=True)
        if patient_signals[signal1].shape != patient_signals[signal2].shape:
            print(f"Signal lengths differ for patient {patient_id}, skipping...")
            continue

        result = calculate_rolling_correlation(
            patient_signals[signal1], patient_signals[signal2], window_size_in_minutes
        )
        result = result.replace([np.inf, -np.inf], np.nan).dropna()

        for cum_window_size in range(
            window_size_in_minutes, max_period_minutes + 1, window_size_in_minutes
        ):
            rolling_values = result[:cum_window_size].values[
                ~np.isnan(result[:cum_window_size].values)
            ]
            if len(rolling_values) == 0:
                continue
            print(rolling_values.shape)
            zcr = calculate_zero_crossing_rate(rolling_values)
            std = calculate_standard_deviation(rolling_values)
            mad = calculate_mean_absolute_deviation(rolling_values)
            rng = calculate_range(rolling_values)
            mean_val = np.mean(rolling_values)
            minus_sign_percentage = calculate_percentage_of_minus_sign(rolling_values)

            hour = cum_window_size // 60
            zcr_data.at[index, f"{signal1}_{signal2}_zcr_{hour}h"] = round(zcr, 3)
            std_data.at[index, f"{signal1}_{signal2}_std_{hour}h"] = round(std, 3)
            mad_data.at[index, f"{signal1}_{signal2}_mad_{hour}h"] = round(mad, 3)
            range_data.at[index, f"{signal1}_{signal2}_range_{hour}h"] = round(rng, 3)
            mean_data.at[index, f"{signal1}_{signal2}_mean_{hour}h"] = round(
                mean_val, 3
            )
            minus_sign_data.at[index, f"{signal1}_{signal2}_minus_{hour}h"] = round(
                minus_sign_percentage, 3
            )

    return zcr_data, std_data, mad_data, range_data, mean_data, minus_sign_data


def perform_correlations(
    metadata_path,
    signals_data_path,
    output_directory,
    ans_signals,
    brain_signals,
    window_days,
    window_hours_list,
):
    metadata = pd.read_csv(metadata_path, index_col=0)
    metadata.dropna(inplace=True)
    model_dir = setup_directories(output_directory, "rolling")

    for window_hours in window_hours_list:
        zcr_df = metadata.copy()
        std_df = metadata.copy()
        mad_df = metadata.copy()
        range_df = metadata.copy()
        mean_df = metadata.copy()
        minus_sign_df = metadata.copy()

        for signal1, signal2 in itertools.product(ans_signals, brain_signals):
            print(f"Processing {signal1} vs {signal2} with window_hours={window_hours}")
            zcr_data, std_data, mad_data, range_data, mean_data, minus_sign_data = (
                calculate_rolling_relationship(
                    metadata,
                    signals_data_path,
                    signal1,
                    signal2,
                    window_days,
                    window_hours,
                )
            )

            zcr_df = zcr_df.join(
                zcr_data.filter(regex="_zcr_"),
                rsuffix=f"_{signal1}_{signal2}",
                how="outer",
            )
            std_df = std_df.join(
                std_data.filter(regex="_std_"),
                rsuffix=f"_{signal1}_{signal2}",
                how="outer",
            )
            mad_df = mad_df.join(
                mad_data.filter(regex="_mad_"),
                rsuffix=f"_{signal1}_{signal2}",
                how="outer",
            )
            range_df = range_df.join(
                range_data.filter(regex="_range_"),
                rsuffix=f"_{signal1}_{signal2}",
                how="outer",
            )
            mean_df = mean_df.join(
                mean_data.filter(regex="_mean_"),
                rsuffix=f"_{signal1}_{signal2}",
                how="outer",
            )
            minus_sign_df = minus_sign_df.join(
                minus_sign_data.filter(regex="_minus_"),
                rsuffix=f"_{signal1}_{signal2}",
                how="outer",
            )

        # zcr_df.to_csv(model_dir / f"zcr_{window_hours}h.csv", float_format="%.3f")
        # std_df.to_csv(model_dir / f"std_{window_hours}h.csv", float_format="%.3f")
        # mad_df.to_csv(model_dir / f"mad_{window_hours}h.csv", float_format="%.3f")
        # range_df.to_csv(model_dir / f"range_{window_hours}h.csv", float_format="%.3f")
        # mean_df.to_csv(model_dir / f"mean_{window_hours}h.csv", float_format="%.3f")
        # minus_sign_df.to_csv(
        #     model_dir / f"minus_{window_hours}h.csv", float_format="%.3f"
        # )


if __name__ == "__main__":
    ans_signals = ["ABP", "ABP_BaroIndex", "HR"]
    brain_signals = ["ICP", "Prx"]
    window_days = 3
    window_hours_list = [3, 6, 12, 24]

    metadata_path = Path("")
    signals_data_path = Path("")
    output_directory = Path("")

    perform_correlations(
        metadata_path,
        signals_data_path,
        output_directory,
        ans_signals,
        brain_signals,
        window_days,
        window_hours_list,
    )

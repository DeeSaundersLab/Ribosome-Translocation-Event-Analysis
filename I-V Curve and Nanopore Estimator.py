"""I-V Curve and Nanopore Estimator module (Enhanced)"""

import os
import csv
import random
from colorsys import rgb_to_hls, hls_to_rgb
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf
from matplotlib.colors import to_hex, to_rgb


# --------------------------------------------
# USER INPUT FOLDER
# --------------------------------------------
INPUT_FOLDER = Path(
    "/Users/deannaellasaunders/Library/CloudStorage/OneDrive-UniversityofLeeds/Ribosome Translocation Experiments/Drosophilia melanogaster/2026-06-11 40S 60S 80S Translocation/Programme 61/IV Curve/50% PEG + 0.1M KCl"
)

if not INPUT_FOLDER.exists():
    raise FileNotFoundError("Input folder does not exist.")


# --------------------------------------------
# PLOT SETTINGS
# --------------------------------------------
SHOW_LEGEND = False


# --------------------------------------------
# ELECTROLYTE DETECTION + COLOUR SYSTEM
# --------------------------------------------
ELECTROLYTE_MAP = {
    "0.1M_KCl": {
        "keywords": ["0.1", "0.1m", "0.1 m"],
        "conductivity": 1.2,
        "color": "#9b59b6",
    },
    "3M_KCl": {
        "keywords": ["3m", "3 m"],
        "conductivity": 25.5,
        "color": "#3498db",
    },
    "PEG_KCl": {
        "keywords": ["peg", "50%", "peg 35k"],
        "conductivity": 1.0,
        "color": "#27ae60",
    },
}


def detect_electrolyte(path_obj):
    path_str = str(path_obj).lower()

    if any(k in path_str for k in ELECTROLYTE_MAP["PEG_KCl"]["keywords"]):
        return "PEG_KCl"
    if any(k in path_str for k in ELECTROLYTE_MAP["3M_KCl"]["keywords"]):
        return "3M_KCl"
    if any(k in path_str for k in ELECTROLYTE_MAP["0.1M_KCl"]["keywords"]):
        return "0.1M_KCl"

    return "0.1M_KCl"


def get_conductivity(electrolyte):
    return ELECTROLYTE_MAP[electrolyte]["conductivity"]


def get_color(electrolyte):
    return ELECTROLYTE_MAP[electrolyte]["color"]


def lighten_color(color, amount=0.45):
    r, g, b = to_rgb(color)
    h, l, s = rgb_to_hls(r, g, b)
    new_l = min(1, l + amount * (1 - l))
    return to_hex(hls_to_rgb(h, new_l, s))


def build_color_shades(base_color, count):
    if count <= 1:
        return [base_color]

    shades = []
    start = 0.08
    stop = 0.5
    for amount in np.linspace(start, stop, count):
        shades.append(lighten_color(base_color, float(amount)))
    return shades


# --------------------------------------------
# OUTPUT FOLDER
# --------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FOLDER = INPUT_FOLDER / f"IV Curve and Nanopore Size Estimator {timestamp}"
CSV_FOLDER = OUTPUT_FOLDER / "CSV"
PLOT_FOLDER = OUTPUT_FOLDER / "Plots"
PLOTTED_FOLDER = PLOT_FOLDER / "Plotted"
LINEAR_REGRESSION_FOLDER = PLOT_FOLDER / "Linear_Regression"

CSV_FOLDER.mkdir(parents=True, exist_ok=True)
PLOTTED_FOLDER.mkdir(parents=True, exist_ok=True)
LINEAR_REGRESSION_FOLDER.mkdir(parents=True, exist_ok=True)

print(f"\nOutput folders created:\n{OUTPUT_FOLDER}\n")


# --------------------------------------------
# ASCII FUN
# --------------------------------------------
animal_ascii = [
    r"""
   /\_/\
  ( o.o )   Cat
   > ^ <
    """,
    r"""
   (\(\
  ( -.-)   Bunny
  o_(")(")
    """,
]

science_one_liners = [
    "Experiment complete—no animals were peer-reviewed.",
    "All variables behaved. Surprisingly.",
    "Zero pipettes cried during execution.",
]


def science_animals_finale():
    print(random.choice(animal_ascii))
    print(">> " + random.choice(science_one_liners))


# --------------------------------------------
# STYLE FUNCTIONS
# --------------------------------------------
def round_down_to_5(value):
    return 5 * np.floor(value / 5)


def round_up_to_5(value):
    return 5 * np.ceil(value / 5)


def set_y_limits_from_data(ax, ydata):
    ymin = float(np.nanmin(ydata))
    ymax = float(np.nanmax(ydata))

    if ymin == ymax:
        ymin -= 5
        ymax += 5

    ymin_lim = 5 * np.floor(ymin / 5)
    ymax_lim = 5 * np.ceil(ymax / 5)

    if ymin_lim == ymax_lim:
        ymin_lim -= 5
        ymax_lim += 5

    ax.set_ylim(ymin_lim, ymax_lim)
    return ymin_lim, ymax_lim


def style_axis(ax, y_limits_for_labels):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.axhline(0, color="grey", linewidth=1)
    ax.axvline(0, color="grey", linewidth=1)

    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.text(xmin, 0, "-500 mV", ha="left", va="bottom", color="grey", fontsize=28, fontweight="bold")
    ax.text(xmax, 0, "+500 mV", ha="right", va="bottom", color="grey", fontsize=28, fontweight="bold")

    ax.text(
        0, ymax, f"{y_limits_for_labels[1]:.0f} pA", ha="left", va="top", color="grey", fontsize=28, fontweight="bold"
    )
    ax.text(
        0, ymin, f"{y_limits_for_labels[0]:.0f} pA", ha="left", va="bottom", color="grey", fontsize=28, fontweight="bold"
    )


def apply_plot_limits(ax, ydata):
    ax.set_xlim(-500, 500)
    y_limits = set_y_limits_from_data(ax, np.asarray(ydata))
    style_axis(ax, y_limits)
    return y_limits


def maybe_add_legend(ax):
    if SHOW_LEGEND:
        ax.legend(frameon=False, fontsize=28)


def get_channel_indices(abf):
    current_idx = None
    voltage_idx = None

    for idx, unit in enumerate(abf.adcUnits):
        unit_lower = unit.lower()
        if unit_lower == "pa" and current_idx is None:
            current_idx = idx
        elif unit_lower == "mv" and voltage_idx is None:
            voltage_idx = idx

    if current_idx is None or voltage_idx is None:
        if abf.channelCount < 2:
            raise ValueError("ABF file does not have separate current and voltage channels.")
        current_idx = 0
        voltage_idx = 1

    return current_idx, voltage_idx


def summarise_by_voltage(df):
    summary = (
        df.groupby("Voltage (mV)", as_index=False)
        .agg(
            current_mean=("Current (pA)", "mean"),
            current_std=("Current (pA)", "std"),
            current_count=("Current (pA)", "count"),
        )
        .sort_values("Voltage (mV)")
    )
    summary["current_sem"] = summary["current_std"] / np.sqrt(summary["current_count"])
    summary["current_sem"] = summary["current_sem"].fillna(0)
    return summary


def fit_linear_regression(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(np.unique(x)) < 2:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept

    return {
        "slope": slope,
        "intercept": intercept,
        "x": x,
        "y_fit": y_fit,
    }


def save_dataframe(df, filename):
    df.to_csv(CSV_FOLDER / filename, index=False)


# --------------------------------------------
# ABF -> CSV
# --------------------------------------------
def convert_abf_files(input_folder, output_folder):
    abf_files = sorted(
        [filename for filename in os.listdir(input_folder) if filename.lower().endswith(".abf")]
    )

    for rep_index, filename in enumerate(abf_files, start=1):
        abf_path = input_folder / filename
        csv_path = output_folder / f"rep{rep_index}.csv"

        abf = pyabf.ABF(str(abf_path))
        current_idx, voltage_idx = get_channel_indices(abf)

        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Sweep", "Time (s)", "Current (pA)", "Voltage (mV)", "Replicate", "Source_File"])

            for sweep in range(abf.sweepCount):
                abf.setSweep(sweep, channel=current_idx)
                time = abf.sweepX
                current = abf.sweepY.copy()

                abf.setSweep(sweep, channel=voltage_idx)
                voltage = abf.sweepY.copy()

                for t, c, v in zip(time, current, voltage):
                    writer.writerow([sweep, t, c, v, f"rep{rep_index}", filename])

        print(f"Converted: {filename} -> {csv_path.name}")

    print("All ABF files converted!\n")


# --------------------------------------------
# IV PLOTS
# --------------------------------------------
def generate_iv_plots(csv_folder, plotted_folder, regression_folder):
    csv_files = sorted(
        [
            path for path in csv_folder.iterdir()
            if path.suffix.lower() == ".csv" and path.name not in {"Nanopore Size.csv"}
        ]
    )

    file_entries = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        if "Voltage (mV)" not in df.columns or "Current (pA)" not in df.columns:
            continue

        df["Voltage (mV)"] = pd.to_numeric(df["Voltage (mV)"], errors="coerce")
        df["Current (pA)"] = pd.to_numeric(df["Current (pA)"], errors="coerce")
        df = df.dropna(subset=["Voltage (mV)", "Current (pA)"])

        if df.empty:
            continue

        electrolyte = detect_electrolyte(csv_path)
        base_color = get_color(electrolyte)
        file_stem = csv_path.stem

        plot_df = df[["Voltage (mV)", "Current (pA)"]].copy()
        plot_df["file"] = csv_path.name
        plot_df["electrolyte"] = electrolyte
        if "Replicate" in df.columns:
            plot_df["Replicate"] = df["Replicate"].values
        if "Source_File" in df.columns:
            plot_df["Source_File"] = df["Source_File"].values
        save_dataframe(plot_df, f"{file_stem}_plotted.csv")

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(plot_df["Voltage (mV)"], plot_df["Current (pA)"], s=8, alpha=0.75, color=base_color)
        apply_plot_limits(ax, plot_df["Current (pA)"])
        fig.savefig(plotted_folder / f"{file_stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        summary = summarise_by_voltage(plot_df)
        summary["file"] = csv_path.name
        summary["electrolyte"] = electrolyte

        regression = fit_linear_regression(summary["Voltage (mV)"], summary["current_mean"])
        if regression is None:
            continue

        regression_df = summary.copy()
        regression_df["fit_current_pA"] = (
            regression["slope"] * regression_df["Voltage (mV)"] + regression["intercept"]
        )
        regression_df["slope_pA_per_mV"] = regression["slope"]
        regression_df["intercept_pA"] = regression["intercept"]
        save_dataframe(regression_df, f"{file_stem}_linear_regression.csv")

        light_color = lighten_color(base_color, 0.55)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(plot_df["Voltage (mV)"], plot_df["Current (pA)"], s=8, alpha=0.35, color=light_color)
        ax.errorbar(
            summary["Voltage (mV)"],
            summary["current_mean"],
            yerr=summary["current_sem"],
            fmt="o",
            markersize=4,
            capsize=3,
            color=base_color,
            ecolor=base_color,
            alpha=0.9,
        )
        ax.plot(summary["Voltage (mV)"], regression_df["fit_current_pA"], color=base_color, linewidth=2.5)
        apply_plot_limits(
            ax,
            np.concatenate(
                [
                    plot_df["Current (pA)"].to_numpy(),
                    (summary["current_mean"] - summary["current_sem"]).to_numpy(),
                    (summary["current_mean"] + summary["current_sem"]).to_numpy(),
                    regression_df["fit_current_pA"].to_numpy(),
                ]
            ),
        )
        fig.savefig(regression_folder / f"{file_stem}_linear_regression.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        entry = {
            "file": csv_path.name,
            "electrolyte": electrolyte,
            "base_color": base_color,
            "plot_df": plot_df,
            "summary_df": summary,
            "regression_df": regression_df,
            "slope": regression["slope"],
            "intercept": regression["intercept"],
        }
        file_entries.append(entry)

    if not file_entries:
        print("No valid CSV files found for plotting.\n")
        return

    replicate_df = pd.concat(
        [entry["summary_df"][["Voltage (mV)", "current_mean", "current_sem", "file", "electrolyte"]] for entry in file_entries],
        ignore_index=True,
    )
    save_dataframe(replicate_df, "IV_replicate_values.csv")

    combined_plot_df = pd.concat(
        [entry["plot_df"] for entry in file_entries],
        ignore_index=True,
    )
    save_dataframe(combined_plot_df, "IV_combined_plotted.csv")

    grouped_by_electrolyte = {}
    for entry in file_entries:
        grouped_by_electrolyte.setdefault(entry["electrolyte"], []).append(entry)

    for electrolyte, entries in grouped_by_electrolyte.items():
        shades = build_color_shades(get_color(electrolyte), len(entries))

        fig, ax = plt.subplots(figsize=(8, 6))
        combined_y = []

        for shade, entry in zip(shades, entries):
            ax.scatter(
                entry["plot_df"]["Voltage (mV)"],
                entry["plot_df"]["Current (pA)"],
                s=8,
                alpha=0.55,
                color=shade,
                label=entry["file"],
            )
            combined_y.extend(entry["plot_df"]["Current (pA)"].tolist())

        apply_plot_limits(ax, np.array(combined_y))
        maybe_add_legend(ax)
        fig.savefig(
            plotted_folder / f"{electrolyte}_combined_plotted.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        combined_stats = (
            pd.concat([entry["summary_df"] for entry in entries], ignore_index=True)
            .groupby("Voltage (mV)", as_index=False)
            .agg(
                current_mean=("current_mean", "mean"),
                current_std=("current_mean", "std"),
                current_count=("current_mean", "count"),
            )
            .sort_values("Voltage (mV)")
        )
        combined_stats["current_sem"] = combined_stats["current_std"] / np.sqrt(combined_stats["current_count"])
        combined_stats["current_sem"] = combined_stats["current_sem"].fillna(0)
        combined_stats["electrolyte"] = electrolyte
        save_dataframe(combined_stats, f"{electrolyte}_combined_mean_sem.csv")

        fig, ax = plt.subplots(figsize=(8, 6))
        mean_sem_y = []
        base_color = get_color(electrolyte)

        for shade, entry in zip(shades, entries):
            ax.plot(
                entry["summary_df"]["Voltage (mV)"],
                entry["summary_df"]["current_mean"],
                color=shade,
                linewidth=1.8,
                alpha=0.95,
                label=entry["file"],
            )
            mean_sem_y.extend(entry["summary_df"]["current_mean"].tolist())

        ax.plot(
            combined_stats["Voltage (mV)"],
            combined_stats["current_mean"],
            color=base_color,
            linewidth=3,
            label=f"{electrolyte} mean",
        )
        ax.fill_between(
            combined_stats["Voltage (mV)"],
            combined_stats["current_mean"] - combined_stats["current_sem"],
            combined_stats["current_mean"] + combined_stats["current_sem"],
            color=base_color,
            alpha=0.2,
        )
        mean_sem_y.extend((combined_stats["current_mean"] - combined_stats["current_sem"]).tolist())
        mean_sem_y.extend((combined_stats["current_mean"] + combined_stats["current_sem"]).tolist())
        apply_plot_limits(ax, np.array(mean_sem_y))
        maybe_add_legend(ax)
        fig.savefig(
            plotted_folder / f"{electrolyte}_combined_mean_sem.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        regression_x = np.linspace(-500, 500, 300)
        regression_rows = []
        fig, ax = plt.subplots(figsize=(8, 6))
        regression_y = []

        for shade, entry in zip(shades, entries):
            fit_y = entry["slope"] * regression_x + entry["intercept"]
            regression_y.extend(fit_y.tolist())
            ax.plot(regression_x, fit_y, color=shade, linewidth=1.8, alpha=0.95, label=entry["file"])

            regression_rows.append(
                pd.DataFrame(
                    {
                        "Voltage (mV)": regression_x,
                        "fit_current_pA": fit_y,
                        "file": entry["file"],
                        "electrolyte": electrolyte,
                        "slope_pA_per_mV": entry["slope"],
                        "intercept_pA": entry["intercept"],
                    }
                )
            )

        regression_matrix = np.vstack(
            [entry["slope"] * regression_x + entry["intercept"] for entry in entries]
        )
        regression_mean = regression_matrix.mean(axis=0)
        regression_sem = (
            regression_matrix.std(axis=0, ddof=1) / np.sqrt(regression_matrix.shape[0])
            if regression_matrix.shape[0] > 1
            else np.zeros_like(regression_mean)
        )

        ax.plot(
            regression_x,
            regression_mean,
            color=base_color,
            linewidth=3,
            label=f"{electrolyte} regression mean",
        )
        ax.fill_between(
            regression_x,
            regression_mean - regression_sem,
            regression_mean + regression_sem,
            color=base_color,
            alpha=0.2,
        )

        regression_y.extend((regression_mean - regression_sem).tolist())
        regression_y.extend((regression_mean + regression_sem).tolist())
        apply_plot_limits(ax, np.array(regression_y))
        maybe_add_legend(ax)
        fig.savefig(
            regression_folder / f"{electrolyte}_combined_linear_regression.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        combined_regression_df = pd.concat(regression_rows, ignore_index=True)
        save_dataframe(combined_regression_df, f"{electrolyte}_combined_linear_regressions.csv")

        combined_regression_summary_df = pd.DataFrame(
            {
                "Voltage (mV)": regression_x,
                "fit_mean_pA": regression_mean,
                "fit_sem_pA": regression_sem,
                "electrolyte": electrolyte,
            }
        )
        save_dataframe(
            combined_regression_summary_df,
            f"{electrolyte}_combined_linear_regression_mean_sem.csv",
        )

    print(f"Plots saved -> {PLOT_FOLDER}\n")


# --------------------------------------------
# RESISTANCE CALCULATION
# --------------------------------------------
def process_csv_folder(folder_path):
    results = []

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(".csv"):
            continue

        if file_name in {
            "Nanopore Size.csv",
            "IV_replicate_values.csv",
            "IV_combined_plotted.csv",
        }:
            continue

        if file_name.endswith("_plotted.csv") or file_name.endswith("_linear_regression.csv"):
            continue

        if "combined_mean_sem" in file_name or "combined_linear_regression" in file_name:
            continue

        file_path = folder_path / file_name
        df = pd.read_csv(file_path)

        if "Voltage (mV)" not in df.columns or "Current (pA)" not in df.columns:
            continue

        pos = df[(df["Voltage (mV)"] >= 50) & (df["Voltage (mV)"] <= 51)]
        neg = df[(df["Voltage (mV)"] <= -50) & (df["Voltage (mV)"] >= -51)]

        if pos.empty or neg.empty:
            continue

        avg_v_pos = pos["Voltage (mV)"].mean()
        avg_v_neg = neg["Voltage (mV)"].mean()
        avg_i_pos = pos["Current (pA)"].mean()
        avg_i_neg = neg["Current (pA)"].mean()

        voltage_diff = avg_v_pos - avg_v_neg
        current_diff = avg_i_pos - avg_i_neg

        if current_diff == 0:
            continue

        resistance = voltage_diff / current_diff * 1000

        results.append({
            "file": file_name,
            "resistance_MOhm": resistance,
        })

    return pd.DataFrame(results)


# --------------------------------------------
# NANOPORE CALC
# --------------------------------------------
def nanopore_radius_estimation(R_MOhm, conductivity=1.2):
    resistance_ohm = R_MOhm * 1e6
    rho = 1 / conductivity
    d_taper = 4e-3
    r_shank = 0.25e-3
    theta = 0.10472

    r_tip = (rho * d_taper) / (np.pi * r_shank * resistance_ohm)
    r_tip_full = (
        1 / (conductivity * np.pi * resistance_ohm * np.tan(theta))
    ) + (
        1 / (4 * conductivity * resistance_ohm)
    )

    return r_tip * 1e9, r_tip_full * 1e9


def compute_nanopore_sizes(summary_df, output_folder):
    results = []

    for _, row in summary_df.iterrows():
        electrolyte = detect_electrolyte(row["file"])
        conductivity = get_conductivity(electrolyte)

        _, r_full = nanopore_radius_estimation(
            row["resistance_MOhm"],
            conductivity,
        )

        results.append({
            "file": row["file"],
            "electrolyte": electrolyte,
            "conductivity": conductivity,
            "resistance_MOhm": row["resistance_MOhm"],
            "r_tip_nm": r_full,
            "diameter_nm": r_full * 2,
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_folder / "Nanopore Size.csv", index=False)
    return output_df


# --------------------------------------------
# MAIN
# --------------------------------------------
def main():
    print("\nRunning nanopore analysis pipeline...\n")
    convert_abf_files(INPUT_FOLDER, CSV_FOLDER)
    generate_iv_plots(CSV_FOLDER, PLOTTED_FOLDER, LINEAR_REGRESSION_FOLDER)
    summary_df = process_csv_folder(CSV_FOLDER)
    if not summary_df.empty:
        compute_nanopore_sizes(summary_df, CSV_FOLDER)
    science_animals_finale()
    print("\nPipeline complete!\n")


if __name__ == "__main__":
    main()

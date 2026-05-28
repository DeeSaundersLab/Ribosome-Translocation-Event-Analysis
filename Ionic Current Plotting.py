# -*- coding: utf-8 -*-
"""
Plots ionic current traces from ABF files.

Features
--------
• Automatically detects sample type from folder path or filename
• Uses identical global y-axis limits for all traces
• Saves all figures into a timestamped output folder
• Generates individual trace plots and a combined trace plot

@author: chalmers
"""

# %%
import pyabf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from pathlib import Path
import os
from datetime import datetime

# -----------------------------
# USER-SPECIFIED FOLDER PATH
# -----------------------------
folder_path = r"/Users/deannaellasaunders/Library/CloudStorage/OneDrive-UniversityofLeeds/Ribosome Translocation Experiments/Misc/Overall Analysis/80S/3.25ng/-700 mV/baseline -1200"

if not os.path.exists(folder_path):
    raise FileNotFoundError("Input folder does not exist.")

abf_files = sorted(Path(folder_path).rglob("*.abf"))

if not abf_files:
    raise FileNotFoundError("No ABF files found in the specified folder.")

# -----------------------------
# OUTPUT FOLDER
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(folder_path, f"Ionic Current Plotting_{timestamp}")
os.makedirs(output_folder, exist_ok=True)

print(f"\nOutput folder created:\n{output_folder}\n")

# -----------------------------
# SUBUNIT COLOR CODING
# -----------------------------
subunit_colors = {
    "RNP": "#e74c3c",
    "40S": "#9b59b6",
    "60S": "#3498db",
    "80S": "#27ae60",
    "40S + 60S": "#c0392b",
    "40S + 60S + 80S": "#1abc9c",
    "40S + 80S": "#f39c12",
    "60S + 80S": "#e67e22",
    "0.1 M KCl": "#566573",
    "Ribosome Buffer": "#d81b60",
    "Mixed Sample": "#ff6f61",
    "Unknown": "#000000"
}

# -----------------------------
# DETECT SUBUNIT FROM PATH
# -----------------------------
def detect_subunit(file_path):

    valid = {
        "40s + 60s + 80s": "40S + 60S + 80S",
        "40s + 60s": "40S + 60S",
        "40s + 80s": "40S + 80S",
        "60s + 80s": "60S + 80S",
        "0.1 m kcl": "0.1 M KCl",
        "ribosome buffer": "Ribosome Buffer",
        "rnp": "RNP",
        "40s": "40S",
        "60s": "60S",
        "80s": "80S",
        "mixed sample": "Mixed Sample"
    }

    # Search path from file backwards
    for p in reversed(file_path.parts):

        key = p.lower()

        for alias, sample_name in valid.items():
            if alias in key:
                return sample_name

    return "Unknown"


# -----------------------------
# EXPORT FIGURE FUNCTION
# -----------------------------
def figure_export_png(folder, name):

    os.makedirs(folder, exist_ok=True)

    save_path = os.path.join(folder, name + ".png")

    plt.savefig(save_path,
                format="png",
                dpi=600,
                bbox_inches="tight")

    print(f"Saved: {save_path}")


# -----------------------------
# LOAD ABF DATA
# -----------------------------
def load_abf_trace(file_path):

    abf = pyabf.ABF(file_path)

    abf.setSweep(0, channel=0)

    raw_unit = abf.adcUnits[0]

    time = abf.sweepX
    current = abf.sweepY

    if raw_unit == "uA":
        raw_unit = "µA"

    unit_factors = {
        "A": 1,
        "mA": 1e-3,
        "µA": 1e-6,
        "nA": 1e-9,
        "pA": 1e-12,
        "fA": 1e-15
    }

    export_unit = "pA"

    factor = unit_factors.get(raw_unit, 1) / unit_factors[export_unit]

    current = current * factor

    df = pd.DataFrame({
        "Time (s)": time,
        "Current (pA)": current
    })

    return df


# -----------------------------
# GLOBAL Y-AXIS LIMITS
# -----------------------------
def get_global_y_limits(files):

    ymin = np.inf
    ymax = -np.inf

    for f in files:

        df = load_abf_trace(f)

        ymin = min(ymin, df["Current (pA)"].min())
        ymax = max(ymax, df["Current (pA)"].max())

    padding = (ymax - ymin) * 0.05

    return ymin - padding, ymax + padding


# -----------------------------
# PLOT SINGLE TRACE
# -----------------------------
def plot_single_trace(df, subunit, ylims):

    sns.set_theme(style="ticks")

    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(df["Time (s)"],
            df["Current (pA)"],
            color=subunit_colors[subunit],
            linewidth=0.5)

    xmax = float(np.max(df["Time (s)"]))

    ax.set_xlim([0, xmax])
    ax.set_ylim(ylims)

    ax.set_xlabel("Time (s)", fontsize=28, fontweight="bold")
    ax.set_ylabel("Current (pA)", fontsize=28, fontweight="bold")

    ax.tick_params(axis="both", labelsize=12)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    plt.tight_layout()

    return fig


# -----------------------------
# PLOT COMBINED TRACES
# -----------------------------
def plot_combined_traces(files, ylims):

    sns.set_theme(style="ticks")

    fig, ax = plt.subplots(figsize=(12, 6))

    used_labels = set()

    for f in files:

        subunit = detect_subunit(f)

        df = load_abf_trace(f)

        label = subunit if subunit not in used_labels else None
        used_labels.add(subunit)

        ax.plot(df["Time (s)"],
                df["Current (pA)"],
                color=subunit_colors[subunit],
                linewidth=0.8,
                label=label)

    ax.set_ylim(ylims)

    ax.set_xlabel("Time (s)", fontsize=28, fontweight="bold")
    ax.set_ylabel("Current (pA)", fontsize=28, fontweight="bold")

    ax.tick_params(axis="both", labelsize=12)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    ax.legend()

    plt.tight_layout()

    return fig


# -----------------------------
# MAIN PROCESSING
# -----------------------------
def main():

    print("Calculating global y-axis limits...")

    ylims = get_global_y_limits(abf_files)

    print(f"Global current range: {ylims}")

    for f in abf_files:

        subunit = detect_subunit(f)

        print(f"Processing {f.name} -> {subunit}")

        df = load_abf_trace(f)

        fig = plot_single_trace(df, subunit, ylims)

        # Prevent overwriting
        figure_export_png(output_folder,
                          f"{f.stem}_{subunit}_trace")

        plt.close(fig)

    print("\nGenerating combined trace plot...")

    fig = plot_combined_traces(abf_files, ylims)

    figure_export_png(output_folder,
                      "combined_traces")

    plt.close(fig)

    print("\nProcessing complete.")


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":

    main()

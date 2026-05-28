#!/usr/bin/env python3

import os
import glob
import ast
import shutil
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from datetime import datetime


# ==========================================================
# INPUT
# ==========================================================

input_folder = "/Users/deannaellasaunders/Library/CloudStorage/OneDrive-UniversityofLeeds/Ribosome Translocation Experiments/Misc/Overall Analysis/80S/3.25ng/-700 mV/baseline -1650"

if not os.path.exists(input_folder):
    raise FileNotFoundError("Input folder does not exist.")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(input_folder, f"Processed_{timestamp}")
os.makedirs(output_folder, exist_ok=True)

csv_folder = os.path.join(output_folder, "CSV")
graph_folder = os.path.join(output_folder, "Graphs")
input_copy_folder = os.path.join(output_folder, "Input_CSVs")

os.makedirs(csv_folder, exist_ok=True)
os.makedirs(graph_folder, exist_ok=True)
os.makedirs(input_copy_folder, exist_ok=True)


# ==========================================================
# METADATA DETECTION
# ==========================================================

folder_parts = input_folder.replace("\\", "/").split("/")

subunit_folder = None
concentration_folder = None
voltage_folder = None

sample_aliases = [
    ("40s + 60s + 80s", "40S + 60S + 80S"),
    ("40s + 60s", "40S + 60S"),
    ("40s + 80s", "40S + 80S"),
    ("60s + 80s", "60S + 80S"),
    ("ribosome buffer", "Ribosome Buffer"),
    ("0.1 m kcl", "0.1 M KCl"),
    ("rnp", "RNP"),
    ("40s", "40S"),
    ("60s", "60S"),
    ("80s", "80S"),
]

for part in reversed(folder_parts):
    part_lower = part.lower()

    if subunit_folder is None:
        for alias, sample_name in sample_aliases:
            if alias in part_lower:
                subunit_folder = sample_name
                break

    if concentration_folder is None and "ng" in part_lower:
        concentration_folder = part

    if voltage_folder is None and "mv" in part_lower:
        voltage_folder = part

    if subunit_folder and concentration_folder and voltage_folder:
        break

if not all([subunit_folder, concentration_folder, voltage_folder]):
    raise ValueError("Could not detect subunit, concentration, or voltage from folder names.")

prefix = f"{subunit_folder}_{concentration_folder}_{voltage_folder}"
voltage_numeric = int(voltage_folder.replace("mV", ""))
concentration_numeric = float(concentration_folder.replace("ng", ""))


# ==========================================================
# COPY ORIGINAL CSVs INTO NEW INPUT FOLDER
# ==========================================================

source_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
if not source_files:
    raise ValueError("No CSV files found.")

copied_files = []
for f in source_files:
    destination = os.path.join(input_copy_folder, os.path.basename(f))
    shutil.copy(f, destination)
    copied_files.append(destination)
    print(f"Copied {os.path.basename(f)} -> {input_copy_folder}")


# ==========================================================
# LOAD + CONCATENATE
# ==========================================================

dfs = []

for rep, f in enumerate(copied_files, start=1):
    try:
        df = pd.read_csv(f)

        df["Replicate"] = rep
        df["Subunit"] = subunit_folder
        df["Concentration (ng)"] = concentration_numeric
        df["Voltage (mV)"] = voltage_numeric

        dfs.append(df)
        print(f"Loaded: {os.path.basename(f)}")

    except Exception as e:
        print(f"Skipping {f} - {e}")

if not dfs:
    raise ValueError("No CSV files could be loaded.")

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv(os.path.join(csv_folder, f"{prefix}_concatenated_input.csv"), index=False)


# ==========================================================
# CLEAN
# ==========================================================

combined_df["Direction"] = combined_df["Direction"].str.lower().str.strip()

def safe_parse(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError, TypeError):
        return (np.nan, np.nan)

combined_df["Parsed_Coordinates"] = combined_df["Coordinates"].apply(safe_parse)
combined_df["Start"] = combined_df["Parsed_Coordinates"].apply(lambda x: x[0])
combined_df["End"] = combined_df["Parsed_Coordinates"].apply(lambda x: x[1])
combined_df["Amplitude (pA)"] = pd.to_numeric(combined_df["Amplitude (pA)"], errors="coerce")
combined_df["Duration (ms)"] = pd.to_numeric(combined_df["Duration (ms)"], errors="coerce")
combined_df["Area (pC)"] = pd.to_numeric(combined_df["Area (pC)"], errors="coerce")

df = combined_df.reset_index(drop=True)
df["Original_Row_Index"] = df.index


# ==========================================================
# FIGURE SAVE HELPER
# ==========================================================

def save_figure_with_padding(path, fig=None):
    if fig is None:
        fig = plt.gcf()

    fig.canvas.draw()
    try:
        fig.tight_layout()
    except Exception:
        pass

    fig.savefig(
        path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.45
    )


# ==========================================================
# BIPHASIC DETECTION
# ==========================================================

biphasic = []
mono = []
unprocessed_biphasic = []

i = 0
biphasic_event_id = 1

while i < len(df):
    cur = df.iloc[i]

    if cur["Direction"] == "up" and i + 1 < len(df):
        nxt = df.iloc[i + 1]

        if nxt["Direction"] == "down" and cur["End"] >= nxt["Start"] - 1:
            merged = cur.copy()
            merged["Amplitude (pA)"] += nxt["Amplitude (pA)"]
            merged["Duration (ms)"] += nxt["Duration (ms)"]
            merged["Area (pC)"] += nxt["Area (pC)"]
            merged["Coordinates"] = (min(cur["Start"], nxt["Start"]), max(cur["End"], nxt["End"]))
            merged["Event_Type"] = "Biphasic"
            merged["Biphasic_Event_ID"] = biphasic_event_id
            biphasic.append(merged)

            cur_unprocessed = cur.copy()
            cur_unprocessed["Event_Type"] = "Biphasic"
            cur_unprocessed["Biphasic_Event_ID"] = biphasic_event_id
            cur_unprocessed["Biphasic_Component"] = "Up"

            nxt_unprocessed = nxt.copy()
            nxt_unprocessed["Event_Type"] = "Biphasic"
            nxt_unprocessed["Biphasic_Event_ID"] = biphasic_event_id
            nxt_unprocessed["Biphasic_Component"] = "Down"

            unprocessed_biphasic.append(cur_unprocessed)
            unprocessed_biphasic.append(nxt_unprocessed)

            biphasic_event_id += 1
            i += 2
            continue

    single = cur.copy()
    single["Event_Type"] = "Monophasic"
    mono.append(single)
    i += 1

biphasic_df = pd.DataFrame(biphasic)
mono_df = pd.DataFrame(mono)
unprocessed_biphasic_df = pd.DataFrame(unprocessed_biphasic)
combined_events = pd.concat([biphasic_df, mono_df], ignore_index=True)


# ==========================================================
# SAVE CSVs
# ==========================================================

combined_events.to_csv(os.path.join(csv_folder, f"{prefix}.csv"), index=False)
biphasic_df.to_csv(os.path.join(csv_folder, f"{prefix}_biphasic.csv"), index=False)
mono_df.to_csv(os.path.join(csv_folder, f"{prefix}_monophasic.csv"), index=False)
unprocessed_biphasic_df.to_csv(
    os.path.join(csv_folder, f"{prefix}_unprocessed_biphasic.csv"),
    index=False
)


# ==========================================================
# PLOTTING SETUP
# ==========================================================

subunit_colors = {
    "RNP": ("#e74c3c", "#922b21"),
    "40S": ("#9b59b6", "#6c3483"),
    "60S": ("#3498db", "#1f618d"),
    "80S": ("#27ae60", "#1e8449"),
    "40S + 60S": ("#c0392b", "#7b241c"),
    "40S + 60S + 80S": ("#1abc9c", "#117864"),
    "40S + 80S": ("#f39c12", "#9a7d0a"),
    "60S + 80S": ("#e67e22", "#af601a"),
    "0.1 M KCl": ("#566573", "#273746"),
    "Ribosome Buffer": ("#d81b60", "#880e4f"),
}

default_colors = ("#7f8c8d", "#2c3e50")

color, accent_color = subunit_colors.get(subunit_folder, default_colors)

plot_df = combined_events[
    (combined_events["Duration (ms)"] >= 0) &
    (combined_events["Duration (ms)"] <= 6) &
    (combined_events["Amplitude (pA)"] >= 0) &
    (combined_events["Amplitude (pA)"] <= 900)
]

mono_plot = plot_df[plot_df["Event_Type"] == "Monophasic"]
bi_plot = plot_df[plot_df["Event_Type"] == "Biphasic"]


# ==========================================================
# CONSISTENT HISTOGRAM BINS
# ==========================================================

x_bin_width = 0.2
y_bin_width = 25

x_bins = np.arange(0, 6 + x_bin_width, x_bin_width)
y_bins = np.arange(0, 900 + y_bin_width, y_bin_width)


# ==========================================================
# JOINT PLOT
# ==========================================================

def joint_plot(data, graph_suffix):
    g = sns.JointGrid(data=data, x="Duration (ms)", y="Amplitude (pA)", height=7)

    g.ax_joint.scatter(
        data["Duration (ms)"],
        data["Amplitude (pA)"],
        s=70,
        alpha=0.35,
        color=color,
        edgecolors="white",
        linewidths=0.5
    )

    sns.histplot(
        data=data,
        x="Duration (ms)",
        bins=x_bins,
        ax=g.ax_marg_x,
        color=color,
        edgecolor="white",
        linewidth=1
    )
    sns.histplot(
        data=data,
        y="Amplitude (pA)",
        bins=y_bins,
        ax=g.ax_marg_y,
        color=color,
        edgecolor="white",
        linewidth=1
    )

    g.ax_joint.set_xlabel("Dwell Time (ms)", fontsize=28, fontweight="bold", color="grey")
    g.ax_joint.set_ylabel("Amplitude (pA)", fontsize=28, fontweight="bold", color="grey")

    g.ax_joint.set_xlim(0, 6)
    g.ax_joint.set_ylim(0, 900)
    g.ax_joint.xaxis.set_major_locator(MultipleLocator(1))
    g.ax_joint.yaxis.set_major_locator(MultipleLocator(100))
    g.ax_joint.tick_params(labelsize=14, colors="grey")

    for spine in g.ax_joint.spines.values():
        spine.set_color("grey")

    g.ax_joint.text(
        0.02,
        0.98,
        f"{subunit_folder}\n n={len(data)}",
        transform=g.ax_joint.transAxes,
        color=accent_color,
        fontsize=18,
        fontweight="bold",
        va="top"
    )

    save_figure_with_padding(
        os.path.join(graph_folder, f"{prefix}_{graph_suffix}.png"),
        fig=g.fig
    )
    plt.close(g.fig)


joint_plot(plot_df, "all_events")
joint_plot(mono_plot, "monophasic_events")
joint_plot(bi_plot, "biphasic_events")


# ==========================================================
# GRID PLOT
# ==========================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

datasets = [plot_df, mono_plot, bi_plot]
titles = ["All Events", "Monophasic Events", "Biphasic Events"]

for idx, (ax, d, t) in enumerate(zip(axes, datasets, titles)):
    ax.scatter(
        d["Duration (ms)"],
        d["Amplitude (pA)"],
        s=50,
        alpha=0.35,
        color=color,
        edgecolors="white",
        linewidths=0.5
    )

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 900)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(100))

    ax_hist_x = ax.inset_axes([0, 1.02, 1, 0.25])
    ax_hist_y = ax.inset_axes([1.02, 0, 0.25, 1])

    sns.histplot(
        data=d,
        x="Duration (ms)",
        bins=x_bins,
        ax=ax_hist_x,
        color=color,
        edgecolor="white",
        linewidth=1
    )
    sns.histplot(
        data=d,
        y="Amplitude (pA)",
        bins=y_bins,
        ax=ax_hist_y,
        color=color,
        edgecolor="white",
        linewidth=1
    )

    ax_hist_x.axis("off")
    ax_hist_y.axis("off")

    if idx == 0:
        ax.set_xlabel("Dwell Time (ms)", fontsize=28, fontweight="bold", color="grey")
        ax.set_ylabel("Amplitude (pA)", fontsize=28, fontweight="bold", color="grey")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    ax.tick_params(labelsize=14, colors="grey")

    for spine in ax.spines.values():
        spine.set_color("grey")

    ax.text(
        0.02,
        0.98,
        f"{subunit_folder}\n n={len(d)}",
        transform=ax.transAxes,
        color=accent_color,
        fontsize=18,
        fontweight="bold",
        va="top"
    )

    ax.set_title(t, fontsize=28, fontweight="bold", color="grey")

save_figure_with_padding(os.path.join(graph_folder, f"{prefix}_grid_events.png"), fig=fig)
plt.close(fig)


# ==========================================================
# COPY SCRIPT
# ==========================================================

shutil.copy(os.path.realpath(__file__), os.path.join(output_folder, os.path.basename(__file__)))


# ==========================================================
# OUTPUT TEXT
# ==========================================================

print("\n🧬 Ribosomes detected — translation unfolding in real time.")
print("⚡ Nanopores capturing single-molecule dynamics.")
print("🔬 Each event encodes structural transitions.")
print("🧪 Biophysics at the ultimate resolution.")
print("📊 Clean data. Clear biology.\n")

print("✅ Pipeline complete.")
print(output_folder)

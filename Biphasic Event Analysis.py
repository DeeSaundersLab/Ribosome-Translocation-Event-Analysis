#!/usr/bin/env python3

import glob
import math
import os
import re
import shutil
import tempfile
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


# ==========================================================
# INPUT
# ==========================================================

default_input_folder = ("/Users/deannaellasaunders/Library/CloudStorage/OneDrive-UniversityofLeeds/Ribosome Translocation Experiments/Misc/Overall Analysis/Biphasic analysis/baseline -1650")
input_folder = os.environ.get("INPUT_FOLDER", default_input_folder)

if not os.path.exists(input_folder):
    raise FileNotFoundError("Input folder does not exist.")

output_base_folder = os.environ.get("OUTPUT_BASE_FOLDER", input_folder)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(output_base_folder, f"Processed_{timestamp}")
os.makedirs(output_folder, exist_ok=True)

csv_folder = os.path.join(output_folder, "CSV")
graph_folder = os.path.join(output_folder, "Graphs")
amplitude_graph_folder = os.path.join(graph_folder, "Amplitude")
coefficient_graph_folder = os.path.join(graph_folder, "Coefficient")
dwell_time_graph_folder = os.path.join(graph_folder, "Dwell Time")
input_copy_folder = os.path.join(output_folder, "Input_CSVs")

os.makedirs(csv_folder, exist_ok=True)
os.makedirs(graph_folder, exist_ok=True)
os.makedirs(input_copy_folder, exist_ok=True)

graph_bucket_names = ["Joint Subunit"]
for root_folder in [amplitude_graph_folder, coefficient_graph_folder, dwell_time_graph_folder]:
    os.makedirs(root_folder, exist_ok=True)
    for subfolder_name in graph_bucket_names:
        os.makedirs(os.path.join(root_folder, subfolder_name), exist_ok=True)


# ==========================================================
# STYLING + DISCOVERY
# ==========================================================

sns.set_style("white")

subunit_order = [
    "RNP",
    "40S",
    "60S",
    "80S",
    "40S + 60S",
    "40S + 60S + 80S",
    "40S + 80S",
    "60S + 80S",
    "0.1 M KCl",
    "Ribosome Buffer",
    "Mixed Sample",
]

DWELL_TIME_BIN_WIDTH = 0.2
AMPLITUDE_BIN_WIDTH = 25
COEFFICIENT_BIN_WIDTH = 0.25
IC_Y_MIN = 1.0
IC_Y_MAX = 2.0
IC_HISTOGRAM_BIN_WIDTH = 0.02
GENERATE_IC_IR_JOINT_SCATTERS = True
SUBUNIT_GRID_COLUMNS = 4

sample_colors = {
    "RNP": ("#ff6b6b", "#7f0000"),
    "40S": ("#9b59b6", "#6c3483"),
    "60S": ("#3498db", "#1f618d"),
    "80S": ("#27ae60", "#1e8449"),
    "40S + 60S": ("#c0392b", "#7b241c"),
    "40S + 60S + 80S": ("#1abc9c", "#117864"),
    "40S + 80S": ("#f39c12", "#9a7d0a"),
    "60S + 80S": ("#e67e22", "#af601a"),
    "0.1 M KCl": ("#566573", "#273746"),
    "Ribosome Buffer": ("#d81b60", "#880e4f"),
    "Mixed Sample": ("#ff6f61", "#c44536"),
}

required_columns = {
    "Amplitude (pA)",
    "Duration (ms)",
    "Biphasic_Event_ID",
    "Biphasic_Component",
}


def sanitize_name(value):
    return str(value).replace(" ", "_")


def build_graph_path(root_folder, bucket_name, filename):
    bucket_folder = os.path.join(root_folder, bucket_name)
    os.makedirs(bucket_folder, exist_ok=True)
    return os.path.join(bucket_folder, filename)


def coefficient_display_label(coefficient_type):
    label_map = {
        "Ic": "I$_c$",
        "Ir": "I$_r$",
        "Dc": "D$_c$",
        "Dr": "D$_r$",
    }
    return label_map.get(coefficient_type, coefficient_type)


def coefficient_slug(coefficient_type):
    return str(coefficient_type).lower()


def histogram_edge_color(subunit):
    return sample_colors.get(subunit, ("#95a5a6", "#34495e"))[1]


def get_present_subunits(df):
    if df is None or df.empty or "Subunit" not in df.columns:
        return []

    present = {
        str(subunit)
        for subunit in df["Subunit"].dropna().tolist()
        if str(subunit).strip()
    }

    ordered = [subunit for subunit in subunit_order if subunit in present]
    extras = sorted(present.difference(subunit_order))
    return ordered + extras


def create_subunit_grid(panel_count, panel_width=6, panel_height=6, sharex=True, sharey=True):
    ncols = min(SUBUNIT_GRID_COLUMNS, max(1, panel_count))
    nrows = math.ceil(panel_count / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    return fig, axes.flatten()


def hide_unused_axes(axes, used_count):
    for ax in axes[used_count:]:
        ax.set_visible(False)


def auto_rotate_x_labels(ax, labels=None, threshold=4, rotation=30):
    if labels is None:
        labels = [tick.get_text() for tick in ax.get_xticklabels()]

    cleaned_labels = [str(label).strip() for label in labels if str(label).strip()]
    if not cleaned_labels:
        return

    should_rotate = len(cleaned_labels) > threshold or max(len(label) for label in cleaned_labels) > 8
    if should_rotate:
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right", rotation_mode="anchor")


def detect_subunit_from_parts(parts):
    labels = [
        "40S + 60S + 80S",
        "40S + 60S",
        "40S + 80S",
        "60S + 80S",
        "0.1 M KCl",
        "Ribosome Buffer",
        "Mixed Sample",
        "RNP",
        "40S",
        "60S",
        "80S",
    ]

    alias_map = [
        ("40s + 60s + 80s", "40S + 60S + 80S"),
        ("40s + 60s", "40S + 60S"),
        ("40s + 80s", "40S + 80S"),
        ("60s + 80s", "60S + 80S"),
        ("0.1 m kcl", "0.1 M KCl"),
        ("ribosome buffer", "Ribosome Buffer"),
        ("mixed sample", "Mixed Sample"),
        ("rnp", "RNP"),
        ("40s", "40S"),
        ("60s", "60S"),
        ("80s", "80S"),
    ]

    for part in parts:
        stripped = str(part).strip()
        lowered = stripped.lower()
        for alias, sample_name in alias_map:
            if alias in lowered:
                return sample_name

    joined = " / ".join(str(part) for part in parts)
    for label in labels:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])", joined, re.IGNORECASE):
            return label

    return "Unknown"


def detect_metadata(file_path):
    normalized = file_path.replace("\\", "/")
    basename = os.path.basename(file_path)
    subunit = detect_subunit_from_parts([basename])

    concentration = None
    voltage = None
    replicate_number = np.nan
    replicate_label = "Rep 1"

    replicate_match = re.search(
        r"(?<![A-Za-z0-9])rep(?:licate)?[\s_-]*(\d+)(?![A-Za-z0-9])",
        normalized,
        re.IGNORECASE,
    )
    if replicate_match:
        replicate_number = int(replicate_match.group(1))
        replicate_label = f"Rep {replicate_number}"

    if concentration is None:
        concentration_match = re.search(r"(\d+(?:\.\d+)?)ng", basename, re.IGNORECASE)
        if concentration_match:
            concentration = f"{concentration_match.group(1)}ng"

    if voltage is None:
        voltage_match = re.search(r"(-?\d+)mV", basename, re.IGNORECASE)
        if voltage_match:
            voltage = f"{voltage_match.group(1)}mV"

    concentration_numeric = (
        float(str(concentration).replace("ng", "")) if concentration is not None else np.nan
    )
    voltage_numeric = int(str(voltage).replace("mV", "")) if voltage is not None else np.nan

    return {
        "Subunit": subunit,
        "Concentration_Label": concentration or "Unknown",
        "Voltage_Label": voltage or "Unknown",
        "Concentration (ng)": concentration_numeric,
        "Voltage (mV)": voltage_numeric,
        "Replicate_Number": replicate_number,
        "Replicate_Label": replicate_label,
    }


def assign_replicate_labels(df):
    if df.empty:
        return df

    assigned_df = df.copy()
    source_metadata = (
        assigned_df[
            [
                "Source_File",
                "Source_Path",
                "Subunit",
                "Concentration (ng)",
                "Voltage (mV)",
                "Replicate_Number",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    source_metadata["Replicate_Number_Resolved"] = source_metadata["Replicate_Number"]

    for subunit, subunit_sources in source_metadata.groupby("Subunit", sort=False):
        missing_mask = subunit_sources["Replicate_Number_Resolved"].isna()
        if not missing_mask.any():
            continue

        existing_numbers = {
            int(value)
            for value in subunit_sources["Replicate_Number_Resolved"].dropna().tolist()
        }
        next_rep = 1

        for index in (
            subunit_sources[missing_mask]
            .sort_values(
                by=["Concentration (ng)", "Voltage (mV)", "Source_File"],
                kind="stable",
            )
            .index
        ):
            while next_rep in existing_numbers:
                next_rep += 1
            source_metadata.loc[index, "Replicate_Number_Resolved"] = next_rep
            existing_numbers.add(next_rep)
            next_rep += 1

    source_metadata["Replicate_Number_Resolved"] = (
        source_metadata["Replicate_Number_Resolved"].fillna(1).astype(int)
    )
    source_metadata["Replicate_Label_Resolved"] = source_metadata["Replicate_Number_Resolved"].map(
        lambda value: f"Rep {value}"
    )

    assigned_df = assigned_df.drop(columns=["Replicate_Number", "Replicate_Label"], errors="ignore")
    assigned_df = assigned_df.merge(
        source_metadata[
            [
                "Source_File",
                "Replicate_Number_Resolved",
                "Replicate_Label_Resolved",
            ]
        ],
        on="Source_File",
        how="left",
    )
    assigned_df = assigned_df.rename(
        columns={
            "Replicate_Number_Resolved": "Replicate_Number",
            "Replicate_Label_Resolved": "Replicate_Label",
        }
    )

    return assigned_df


def discover_candidate_csvs(root_folder):
    candidates = sorted(glob.glob(os.path.join(root_folder, "**", "*.csv"), recursive=True))
    valid = []

    for path in candidates:
        normalized = path.replace("\\", "/")
        if "/Processed_" in normalized:
            continue

        try:
            preview = pd.read_csv(path, nrows=3)
        except Exception as exc:
            print(f"Skipping unreadable CSV: {path} ({exc})")
            continue

        if required_columns.issubset(preview.columns):
            valid.append(path)

    return valid


def load_biphasic_components(file_path):
    metadata = detect_metadata(file_path)
    df = pd.read_csv(file_path).copy()

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {file_path}")

    df["Amplitude (pA)"] = pd.to_numeric(df["Amplitude (pA)"], errors="coerce")
    df["Duration (ms)"] = pd.to_numeric(df["Duration (ms)"], errors="coerce")
    df["Biphasic_Event_ID"] = pd.to_numeric(df["Biphasic_Event_ID"], errors="coerce")
    df["Biphasic_Component"] = df["Biphasic_Component"].astype(str).str.strip().str.lower()

    if "Subunit" in df.columns:
        csv_subunit = (
            df["Subunit"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        if not csv_subunit.empty:
            metadata["Subunit"] = detect_subunit_from_parts(csv_subunit.tolist() + [metadata["Subunit"]])

    df["Source_File"] = os.path.basename(file_path)
    df["Source_Path"] = file_path
    df["Subunit"] = metadata["Subunit"]
    df["Concentration (ng)"] = metadata["Concentration (ng)"]
    df["Voltage (mV)"] = metadata["Voltage (mV)"]
    df["Concentration_Label"] = metadata["Concentration_Label"]
    df["Voltage_Label"] = metadata["Voltage_Label"]
    df["Replicate_Number"] = metadata["Replicate_Number"]
    df["Replicate_Label"] = metadata["Replicate_Label"]

    return df


def build_component_and_coefficient_tables(raw_df):
    components = []
    coefficients = []
    dwell_coefficients = []

    grouped = raw_df.groupby(["Source_File", "Subunit", "Biphasic_Event_ID"], dropna=True)

    for (source_file, subunit, event_id), group in grouped:
        if pd.isna(event_id):
            continue

        up_rows = group[group["Biphasic_Component"] == "up"]
        down_rows = group[group["Biphasic_Component"] == "down"]

        if up_rows.empty or down_rows.empty:
            continue

        up = up_rows.iloc[0]
        down = down_rows.iloc[0]

        up_amp = up["Amplitude (pA)"]
        down_amp = down["Amplitude (pA)"]
        up_duration = up["Duration (ms)"]
        down_duration = down["Duration (ms)"]
        total_amp = up_amp + down_amp
        total_dwell = up_duration + down_duration

        shared = {
            "Source_File": source_file,
            "Subunit": subunit,
            "Biphasic_Event_ID": int(event_id),
            "Concentration (ng)": up["Concentration (ng)"],
            "Voltage (mV)": up["Voltage (mV)"],
            "Replicate_Number": up["Replicate_Number"],
            "Replicate_Label": up["Replicate_Label"],
            "Total Biphasic Amplitude (pA)": total_amp,
            "Total Biphasic Dwell Time (ms)": total_dwell,
        }

        components.append(
            {
                **shared,
                "Direction": "up",
                "Amplitude (pA)": up_amp,
                "Dwell Time (ms)": up_duration,
            }
        )
        components.append(
            {
                **shared,
                "Direction": "down",
                "Amplitude (pA)": down_amp,
                "Dwell Time (ms)": down_duration,
            }
        )

        if pd.notna(down_amp) and down_amp != 0 and pd.notna(down_duration):
            coefficients.append(
                {
                    **shared,
                    "Coefficient_Type": "Ic",
                    "Coefficient": total_amp / down_amp,
                    "Direction": "down",
                    "Dwell Time (ms)": down_duration,
                }
            )

        if pd.notna(up_amp) and up_amp != 0 and pd.notna(up_duration):
            coefficients.append(
                {
                    **shared,
                    "Coefficient_Type": "Ir",
                    "Coefficient": total_amp / up_amp,
                    "Direction": "up",
                    "Dwell Time (ms)": up_duration,
                }
            )

        if pd.notna(down_duration) and down_duration != 0 and pd.notna(total_dwell):
            dwell_coefficients.append(
                {
                    **shared,
                    "Coefficient_Type": "Dc",
                    "Coefficient": total_dwell / down_duration,
                    "Direction": "down",
                    "Dwell Time (ms)": down_duration,
                }
            )

        if pd.notna(up_duration) and up_duration != 0 and pd.notna(total_dwell):
            dwell_coefficients.append(
                {
                    **shared,
                    "Coefficient_Type": "Dr",
                    "Coefficient": total_dwell / up_duration,
                    "Direction": "up",
                    "Dwell Time (ms)": up_duration,
                }
            )

    component_df = pd.DataFrame(components)
    coefficient_df = pd.DataFrame(coefficients)
    dwell_coefficient_df = pd.DataFrame(dwell_coefficients)

    return component_df, coefficient_df, dwell_coefficient_df


def clean_plot_df(df, x_col, y_col, x_max=None, y_max=None):
    plot_df = df.copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=[x_col, y_col, "Subunit"])
    plot_df = plot_df[(plot_df[x_col] >= 0) & (plot_df[y_col] >= 0)]

    if x_max is not None:
        plot_df = plot_df[plot_df[x_col] <= x_max]
    if y_max is not None:
        plot_df = plot_df[plot_df[y_col] <= y_max]

    return plot_df


def nice_upper_limit(series, default_value):
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[clean >= 0]

    if clean.empty:
        return default_value

    percentile = clean.quantile(0.99)
    candidate = max(percentile * 1.05, clean.max() * 1.01)

    if candidate <= 0:
        return default_value

    magnitude = 10 ** math.floor(math.log10(candidate)) if candidate > 0 else 1
    rounded = math.ceil(candidate / magnitude * 2) / 2 * magnitude
    return max(default_value, rounded)


def x_bins_from_limit(x_limit, step):
    return np.arange(0, x_limit + step, step)


def y_bins_from_limit(y_limit, step, y_min=0):
    return np.arange(y_min, y_limit + step, step)


def style_axes(ax):
    ax.tick_params(labelsize=16, colors="grey")
    ax.xaxis.label.set_color("grey")
    ax.yaxis.label.set_color("grey")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("grey")
    ax.spines["bottom"].set_color("grey")


def add_panel_label(ax, subunit, n_value, accent_color, side="left", title_text=None):
    x_pos = 0.02 if side == "left" else 0.98
    ha = "left" if side == "left" else "right"
    panel_title = title_text or subunit
    ax.text(
        x_pos,
        0.98,
        f"{panel_title}\nn={n_value}",
        transform=ax.transAxes,
        color=accent_color,
        fontsize=20,
        fontweight="bold",
        va="top",
        ha=ha,
    )


def add_direction_legend(ax, light_color, dark_color, up_label="Up", down_label="Down"):
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=light_color, markersize=8, label=up_label),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=dark_color, markersize=8, label=down_label),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=11)


def add_single_color_legend(ax, color, label):
    handle = Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label)
    ax.legend(handles=[handle], loc="lower right", frameon=False, fontsize=11)


def style_legend_text(legend, text_color=None):
    if legend is None or text_color is None:
        return
    for text in legend.get_texts():
        text.set_color(text_color)
    if legend.get_title() is not None:
        legend.get_title().set_color(text_color)


def plot_joint_component(
    subunit_df,
    output_path,
    y_label,
    y_col,
    x_col,
    title_suffix,
    x_limit,
    y_min,
    y_limit,
    y_bins,
    use_direction_shading=True,
    color=None,
    legend_label=None,
    show_legend=True,
    up_label="Up",
    down_label="Down",
):
    subunit = subunit_df["Subunit"].iloc[0]
    light_color, dark_color = sample_colors.get(subunit, ("#95a5a6", "#34495e"))
    x_bins = x_bins_from_limit(x_limit, DWELL_TIME_BIN_WIDTH)

    g = sns.JointGrid(data=subunit_df, x=x_col, y=y_col, height=8)

    if use_direction_shading:
        for direction, direction_color in [("up", light_color), ("down", dark_color)]:
            direction_df = subunit_df[subunit_df["Direction"] == direction]
            if direction_df.empty:
                continue

            g.ax_joint.scatter(
                direction_df[x_col],
                direction_df[y_col],
                s=50,
                alpha=0.35,
                color=direction_color,
            )

            sns.histplot(
                data=direction_df,
                x=x_col,
                bins=x_bins,
                ax=g.ax_marg_x,
                color=direction_color,
                alpha=0.55,
                element="bars",
            )
            sns.histplot(
                data=direction_df,
                y=y_col,
                bins=y_bins,
                ax=g.ax_marg_y,
                color=direction_color,
                alpha=0.55,
                element="bars",
            )
    else:
        plot_color = color or dark_color

        g.ax_joint.scatter(
            subunit_df[x_col],
            subunit_df[y_col],
            s=50,
            alpha=0.35,
            color=plot_color,
        )

        sns.histplot(
            data=subunit_df,
            x=x_col,
            bins=x_bins,
            ax=g.ax_marg_x,
            color=plot_color,
            alpha=0.55,
            element="bars",
        )
        sns.histplot(
            data=subunit_df,
            y=y_col,
            bins=y_bins,
            ax=g.ax_marg_y,
            color=plot_color,
            alpha=0.55,
            element="bars",
        )

    g.ax_joint.set_xlabel("Dwell Time (ms)", fontsize=28, fontweight="bold", color="grey")
    g.ax_joint.set_ylabel(y_label, fontsize=28, fontweight="bold", color="grey")
    g.ax_joint.set_xlim(0, x_limit)
    g.ax_joint.set_ylim(y_min, y_limit)
    g.ax_joint.xaxis.set_major_locator(MultipleLocator(1))
    style_axes(g.ax_joint)
    add_panel_label(g.ax_joint, subunit, len(subunit_df), dark_color)
    if use_direction_shading and show_legend:
        add_direction_legend(g.ax_joint, light_color, dark_color, up_label=up_label, down_label=down_label)
    elif (not use_direction_shading) and show_legend:
        add_single_color_legend(g.ax_joint, color or dark_color, legend_label or title_suffix)
    g.ax_joint.set_title(title_suffix, fontsize=28, fontweight="bold", color="grey", pad=16)

    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_combined_coefficient_joint_grid(plot_df, output_path, x_limit, y_min, y_limit, y_bins, title, coefficient_types):
    active_subunits = get_present_subunits(plot_df)
    if not active_subunits:
        return

    x_bins = x_bins_from_limit(x_limit, DWELL_TIME_BIN_WIDTH)
    fig, axes = create_subunit_grid(len(active_subunits), sharex=True, sharey=True)

    for idx, subunit in enumerate(active_subunits):
        ax = axes[idx]
        subunit_df = plot_df[plot_df["Subunit"] == subunit]
        light_color, dark_color = sample_colors.get(subunit, ("#95a5a6", "#34495e"))

        if not subunit_df.empty:
            for coefficient_type in coefficient_types:
                coefficient_type_df = subunit_df[subunit_df["Coefficient_Type"] == coefficient_type]
                if coefficient_type_df.empty:
                    continue
                color = light_color if coefficient_type.endswith("r") else dark_color

                ax.scatter(
                    coefficient_type_df["Dwell Time (ms)"],
                    coefficient_type_df["Coefficient"],
                    s=50,
                    alpha=0.35,
                    color=color,
                    label=coefficient_display_label(coefficient_type),
                )

                ax_hist_x = ax.inset_axes([0, 1.02, 1, 0.25])
                ax_hist_y = ax.inset_axes([1.02, 0, 0.25, 1])

                sns.histplot(
                    data=coefficient_type_df,
                    x="Dwell Time (ms)",
                    bins=x_bins,
                    ax=ax_hist_x,
                    color=color,
                    alpha=0.55,
                    element="bars",
                )
                sns.histplot(
                    data=coefficient_type_df,
                    y="Coefficient",
                    bins=y_bins,
                    ax=ax_hist_y,
                    color=color,
                    alpha=0.55,
                    element="bars",
                )

                ax_hist_x.axis("off")
                ax_hist_y.axis("off")

        ax.set_xlim(0, x_limit)
        ax.set_ylim(y_min, y_limit)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        style_axes(ax)
        add_panel_label(
            ax,
            subunit,
            len(subunit_df),
            dark_color,
            side="right",
            title_text=subunit,
        )
        if not subunit_df.empty:
            legend = ax.legend(loc="upper right", frameon=False, fontsize=11, title="")
            style_legend_text(legend, text_color="grey")
        ax.set_title("")

        ax.set_xlabel("Dwell Time (ms)", fontsize=28, fontweight="bold", color="grey")
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_ylabel("Coefficient" if idx == 0 else "", fontsize=28, fontweight="bold", color="grey")

    hide_unused_axes(axes, len(active_subunits))
    fig.suptitle(title, fontsize=28, fontweight="bold", color="grey", y=1.06)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_overlay_joint_by_subunit(
    plot_df,
    output_path,
    coefficient_type,
    title_suffix,
    x_limit,
    y_min,
    y_limit,
    y_bins,
    show_legend=True,
):
    x_bins = x_bins_from_limit(x_limit, DWELL_TIME_BIN_WIDTH)
    coefficient_type_df = plot_df[plot_df["Coefficient_Type"] == coefficient_type]

    if coefficient_type_df.empty:
        return

    g = sns.JointGrid(data=coefficient_type_df, x="Dwell Time (ms)", y="Coefficient", height=8)
    legend_handles = []

    for subunit in subunit_order:
        subunit_df = coefficient_type_df[coefficient_type_df["Subunit"] == subunit]
        if subunit_df.empty:
            continue

        color = sample_colors.get(subunit, ("#95a5a6", "#34495e"))[0]
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=subunit)
        )

        g.ax_joint.scatter(
            subunit_df["Dwell Time (ms)"],
            subunit_df["Coefficient"],
            s=70,
            alpha=0.35,
            color=color,
        )

        sns.histplot(
            data=subunit_df,
            x="Dwell Time (ms)",
            bins=x_bins,
            ax=g.ax_marg_x,
            color=color,
            alpha=0.40,
            element="bars",
        )
        sns.histplot(
            data=subunit_df,
            y="Coefficient",
            bins=y_bins,
            ax=g.ax_marg_y,
            color=color,
            alpha=0.40,
            element="bars",
        )

    g.ax_joint.set_xlabel("Dwell Time (ms)", fontsize=28, fontweight="bold", color="grey")
    g.ax_joint.set_ylabel(f"{coefficient_display_label(coefficient_type)} Coefficient", fontsize=28, fontweight="bold", color="grey")
    g.ax_joint.set_xlim(0, x_limit)
    g.ax_joint.set_ylim(y_min, y_limit)
    g.ax_joint.xaxis.set_major_locator(MultipleLocator(1))
    style_axes(g.ax_joint)
    if show_legend:
        g.ax_joint.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=11, title="")
    g.ax_joint.set_title(title_suffix, fontsize=28, fontweight="bold", color="grey", pad=16)

    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_overlay_coefficient_grid(
    plot_df,
    output_path,
    x_limit,
    y_mins,
    y_limits,
    y_bins_map,
    title,
    coefficient_types,
    label_ir_axes=False,
):
    x_bins = x_bins_from_limit(x_limit, DWELL_TIME_BIN_WIDTH)
    fig, axes = plt.subplots(1, len(coefficient_types), figsize=(6 * len(coefficient_types), 6), sharex=True, sharey=False)

    if len(coefficient_types) == 1:
        axes = [axes]

    for ax, coefficient_type in zip(axes, coefficient_types):
        type_df = plot_df[plot_df["Coefficient_Type"] == coefficient_type]

        for subunit in subunit_order:
            subunit_df = type_df[type_df["Subunit"] == subunit]
            if subunit_df.empty:
                continue

            color = sample_colors.get(subunit, ("#95a5a6", "#34495e"))[0]
            ax.scatter(
                subunit_df["Dwell Time (ms)"],
                subunit_df["Coefficient"],
                s=50,
                alpha=0.35,
                color=color,
                label=subunit,
            )

            ax_hist_x = ax.inset_axes([0, 1.02, 1, 0.25])
            ax_hist_y = ax.inset_axes([1.02, 0, 0.25, 1])

            sns.histplot(
                data=subunit_df,
                x="Dwell Time (ms)",
                bins=x_bins,
                ax=ax_hist_x,
                color=color,
                alpha=0.40,
                element="bars",
            )
            sns.histplot(
                data=subunit_df,
                y="Coefficient",
                bins=y_bins_map[coefficient_type],
                ax=ax_hist_y,
                color=color,
                alpha=0.40,
                element="bars",
            )

            ax_hist_x.axis("off")
            ax_hist_y.axis("off")

        ax.set_xlim(0, x_limit)
        ax.set_ylim(y_mins[coefficient_type], y_limits[coefficient_type])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        style_axes(ax)
        legend = ax.legend(loc="lower right", frameon=False, fontsize=11, title="")
        style_legend_text(legend, text_color="grey")
        ax.set_title(coefficient_display_label(coefficient_type), fontsize=28, fontweight="bold", color="grey")
        ax.set_xlabel("Dwell Time (ms)", fontsize=28, fontweight="bold", color="grey")

        if coefficient_type in {"Ic", "Dc"} or label_ir_axes:
            ax.set_ylabel("Coefficient", fontsize=28, fontweight="bold", color="grey")
        else:
            ax.set_ylabel("")

    fig.suptitle(title, fontsize=28, fontweight="bold", color="grey", y=1.06)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_three_panel_grid(
    plot_df,
    output_path,
    y_label,
    y_col,
    x_col,
    x_limit,
    y_min,
    y_limit,
    y_bins,
    title,
    use_direction_shading=True,
    legend_label=None,
    show_legend=True,
    show_subplot_titles=True,
    panel_label_side="left",
    up_label="Up",
    down_label="Down",
    single_color=None,
):
    active_subunits = get_present_subunits(plot_df)
    if not active_subunits:
        return

    x_bins = x_bins_from_limit(x_limit, DWELL_TIME_BIN_WIDTH)
    fig, axes = create_subunit_grid(len(active_subunits), sharex=True, sharey=True)

    for idx, subunit in enumerate(active_subunits):
        ax = axes[idx]
        subunit_df = plot_df[plot_df["Subunit"] == subunit]
        light_color, dark_color = sample_colors.get(subunit, ("#95a5a6", "#34495e"))

        if use_direction_shading:
            for direction, color in [("up", light_color), ("down", dark_color)]:
                direction_df = subunit_df[subunit_df["Direction"] == direction]
                if direction_df.empty:
                    continue

                ax.scatter(
                    direction_df[x_col],
                    direction_df[y_col],
                    s=50,
                    alpha=0.35,
                    color=color,
                )

                ax_hist_x = ax.inset_axes([0, 1.02, 1, 0.25])
                ax_hist_y = ax.inset_axes([1.02, 0, 0.25, 1])

                sns.histplot(
                    data=direction_df,
                    x=x_col,
                    bins=x_bins,
                    ax=ax_hist_x,
                    color=color,
                    alpha=0.55,
                    element="bars",
                )
                sns.histplot(
                    data=direction_df,
                    y=y_col,
                    bins=y_bins,
                    ax=ax_hist_y,
                    color=color,
                    alpha=0.55,
                    element="bars",
                )

                ax_hist_x.axis("off")
                ax_hist_y.axis("off")
        elif not subunit_df.empty:
            panel_color = single_color or light_color

            ax.scatter(
                subunit_df[x_col],
                subunit_df[y_col],
                s=50,
                alpha=0.35,
                color=panel_color,
            )

            ax_hist_x = ax.inset_axes([0, 1.02, 1, 0.25])
            ax_hist_y = ax.inset_axes([1.02, 0, 0.25, 1])

            sns.histplot(
                data=subunit_df,
                x=x_col,
                bins=x_bins,
                ax=ax_hist_x,
                color=panel_color,
                alpha=0.55,
                element="bars",
            )
            sns.histplot(
                data=subunit_df,
                y=y_col,
                bins=y_bins,
                ax=ax_hist_y,
                color=panel_color,
                alpha=0.55,
                element="bars",
            )

            ax_hist_x.axis("off")
            ax_hist_y.axis("off")

        ax.set_xlim(0, x_limit)
        ax.set_ylim(y_min, y_limit)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        style_axes(ax)
        label_color = single_color or light_color
        add_panel_label(ax, subunit, len(subunit_df), label_color, side=panel_label_side)
        if use_direction_shading and show_legend:
            add_direction_legend(ax, light_color, dark_color, up_label=up_label, down_label=down_label)
        elif (not use_direction_shading) and show_legend and (not subunit_df.empty):
            add_single_color_legend(ax, single_color or light_color, legend_label or y_label)
        ax.set_title(subunit if show_subplot_titles else "", fontsize=28, fontweight="bold", color="grey")

        ax.set_xlabel("Dwell Time (ms)", fontsize=28, fontweight="bold", color="grey")
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_ylabel(y_label if idx == 0 else "", fontsize=28, fontweight="bold", color="grey")

    hide_unused_axes(axes, len(active_subunits))
    fig.suptitle(title, fontsize=28, fontweight="bold", color="grey", y=1.06)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def get_replicate_order(df):
    if df.empty or "Replicate_Number" not in df.columns:
        return []

    replicate_numbers = (
        pd.Series(df["Replicate_Number"])
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
    return [f"Rep {value}" for value in replicate_numbers]


def draw_coefficient_boxplot(ax, panel_df, x_col, order, palette, y_label, title, y_min, y_limit):
    if panel_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14, color="grey")
        ax.set_title(title, fontsize=28, fontweight="bold", color="grey")
        ax.set_xlabel(x_col.replace("_", " "), fontsize=28, fontweight="bold", color="grey")
        ax.set_ylabel(y_label, fontsize=28, fontweight="bold", color="grey")
        ax.set_ylim(y_min, y_limit)
        style_axes(ax)
        auto_rotate_x_labels(ax, order)
        return

    sns.boxplot(
        data=panel_df,
        x=x_col,
        y="Coefficient",
        order=order,
        hue=x_col,
        hue_order=order,
        dodge=False,
        palette=palette,
        width=0.55,
        fliersize=0,
        linewidth=1.5,
        boxprops=dict(facecolor="none", edgecolor="grey"),
        whiskerprops=dict(color="grey"),
        capprops=dict(color="grey"),
        medianprops=dict(color="none", linewidth=0),
        ax=ax,
    )
    sns.stripplot(
        data=panel_df,
        x=x_col,
        y="Coefficient",
        order=order,
        hue=x_col,
        hue_order=order,
        dodge=False,
        palette=palette,
        alpha=0.35,
        size=5,
        jitter=0.18,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()

    ax.set_xlabel(x_col.replace("_", " "), fontsize=28, fontweight="bold", color="grey")
    ax.set_ylabel(y_label, fontsize=28, fontweight="bold", color="grey")
    ax.set_title(title, fontsize=28, fontweight="bold", color="grey")
    ax.set_ylim(y_min, y_limit)
    style_axes(ax)
    auto_rotate_x_labels(ax, order)


def draw_coefficient_violinplot(ax, panel_df, x_col, order, palette, y_label, title, y_min, y_limit):
    if panel_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14, color="grey")
        ax.set_title(title, fontsize=28, fontweight="bold", color="grey")
        ax.set_xlabel(x_col.replace("_", " "), fontsize=28, fontweight="bold", color="grey")
        ax.set_ylabel(y_label, fontsize=28, fontweight="bold", color="grey")
        ax.set_ylim(y_min, y_limit)
        style_axes(ax)
        auto_rotate_x_labels(ax, order)
        return

    sns.violinplot(
        data=panel_df,
        x=x_col,
        y="Coefficient",
        order=order,
        hue=x_col,
        hue_order=order,
        dodge=False,
        palette=palette,
        cut=0,
        inner=None,
        linewidth=1.2,
        saturation=1,
        ax=ax,
    )
    sns.stripplot(
        data=panel_df,
        x=x_col,
        y="Coefficient",
        order=order,
        hue=x_col,
        hue_order=order,
        dodge=False,
        palette=palette,
        alpha=0.3,
        size=4.5,
        jitter=0.18,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()

    ax.set_xlabel(x_col.replace("_", " "), fontsize=28, fontweight="bold", color="grey")
    ax.set_ylabel(y_label, fontsize=28, fontweight="bold", color="grey")
    ax.set_title(title, fontsize=28, fontweight="bold", color="grey")
    ax.set_ylim(y_min, y_limit)
    style_axes(ax)
    auto_rotate_x_labels(ax, order)


def build_replicate_palette(subunit, replicate_order):
    base_color = sample_colors.get(subunit, ("#95a5a6", "#34495e"))[0]
    return {replicate_label: base_color for replicate_label in replicate_order}


def plot_coefficient_histograms(coefficient_df, output_root, coefficient_type, x_min, x_max, bin_width):
    active_subunits = get_present_subunits(coefficient_df)
    if not active_subunits:
        return

    coefficient_label = coefficient_display_label(coefficient_type)
    bins = np.arange(x_min, x_max + bin_width, bin_width)

    for subunit in active_subunits:
        subunit_df = coefficient_df[
            (coefficient_df["Subunit"] == subunit) & (coefficient_df["Coefficient_Type"] == coefficient_type)
        ]
        if subunit_df.empty:
            continue

        color = sample_colors.get(subunit, ("#95a5a6", "#34495e"))[0]
        edge_color = histogram_edge_color(subunit)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(
            data=subunit_df,
            x="Coefficient",
            bins=bins,
            color=color,
            alpha=0.45,
            element="bars",
            edgecolor=edge_color,
            linewidth=1.0,
            ax=ax,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(f"{coefficient_label} Coefficient", fontsize=28, fontweight="bold", color="grey")
        ax.set_ylabel("Number of Events", fontsize=28, fontweight="bold", color="grey")
        ax.set_title(f"{subunit} {coefficient_label} Histogram", fontsize=28, fontweight="bold", color="grey")
        style_axes(ax)
        plt.tight_layout()
        plt.savefig(
            build_graph_path(output_root, subunit, f"{subunit}_{coefficient_slug(coefficient_type)}_histogram.png"),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    fig, axes = create_subunit_grid(len(active_subunits), sharex=True, sharey=True)
    for idx, subunit in enumerate(active_subunits):
        ax = axes[idx]
        subunit_df = coefficient_df[
            (coefficient_df["Subunit"] == subunit) & (coefficient_df["Coefficient_Type"] == coefficient_type)
        ]
        color = sample_colors.get(subunit, ("#95a5a6", "#34495e"))[0]

        if subunit_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14, color="grey")
        else:
            edge_color = histogram_edge_color(subunit)
            sns.histplot(
                data=subunit_df,
                x="Coefficient",
                bins=bins,
                color=color,
                alpha=0.45,
                element="bars",
                edgecolor=edge_color,
                linewidth=1.0,
                ax=ax,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_title(subunit, fontsize=28, fontweight="bold", color="grey")
        ax.set_xlabel(f"{coefficient_label} Coefficient", fontsize=28, fontweight="bold", color="grey")
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_ylabel("Number of Events" if idx == 0 else "", fontsize=28, fontweight="bold", color="grey")
        style_axes(ax)

    hide_unused_axes(axes, len(active_subunits))
    fig.suptitle(f"{coefficient_label} Histograms", fontsize=28, fontweight="bold", color="grey", y=1.02)
    plt.tight_layout()
    plt.savefig(
        build_graph_path(output_root, "Joint Subunit", f"40S_60S_80S_{coefficient_slug(coefficient_type)}_histogram_grid.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))
    legend_handles = []

    for subunit in active_subunits:
        subunit_df = coefficient_df[
            (coefficient_df["Subunit"] == subunit) & (coefficient_df["Coefficient_Type"] == coefficient_type)
        ]
        if subunit_df.empty:
            continue

        color = sample_colors.get(subunit, ("#95a5a6", "#34495e"))[0]
        edge_color = histogram_edge_color(subunit)
        sns.histplot(
            data=subunit_df,
            x="Coefficient",
            bins=bins,
            color=color,
            alpha=0.22,
            element="bars",
            stat="count",
            common_norm=False,
            edgecolor=edge_color,
            linewidth=1.15,
            ax=ax,
        )
        legend_handles.append(
            Line2D([0], [0], marker="s", color="w", markerfacecolor=color, alpha=0.55, markersize=10, label=subunit)
        )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(f"{coefficient_label} Coefficient", fontsize=28, fontweight="bold", color="grey")
    ax.set_ylabel("Number of Events", fontsize=28, fontweight="bold", color="grey")
    ax.set_title(f"Combined {coefficient_label} Histogram", fontsize=28, fontweight="bold", color="grey")
    style_axes(ax)
    if legend_handles:
        legend = ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=11, title="")
        style_legend_text(legend, text_color="grey")
    plt.tight_layout()
    plt.savefig(
        build_graph_path(
            output_root,
            "Joint Subunit",
            f"all_subunits_{coefficient_slug(coefficient_type)}_histogram_overlay.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


def plot_boxplots(coefficient_df, output_root, coefficient_types, y_mins, y_limits, filename_prefix, summary_filename):
    if coefficient_df.empty:
        return

    box_df = coefficient_df.copy()
    active_subunits = get_present_subunits(box_df)
    if not active_subunits:
        return

    box_df["Subunit"] = pd.Categorical(box_df["Subunit"], categories=active_subunits, ordered=True)
    box_df["Coefficient_Type"] = pd.Categorical(box_df["Coefficient_Type"], categories=coefficient_types, ordered=True)
    replicate_order = get_replicate_order(box_df)
    if replicate_order:
        box_df["Replicate_Label"] = pd.Categorical(
            box_df["Replicate_Label"],
            categories=replicate_order,
            ordered=True,
        )
    box_df = box_df.sort_values(["Subunit", "Coefficient_Type"])

    summary = (
        box_df.groupby(["Subunit", "Replicate_Label", "Coefficient_Type"], observed=True)["Coefficient"]
        .agg(["count", "mean", "min", "max", "median", "std"])
        .reset_index()
    )
    summary.to_csv(os.path.join(csv_folder, summary_filename), index=False)

    fig, axes = plt.subplots(1, len(coefficient_types), figsize=(7 * len(coefficient_types), 6), sharey=False)
    if len(coefficient_types) == 1:
        axes = [axes]

    for ax, coefficient_type in zip(axes, coefficient_types):
        panel_df = box_df[box_df["Coefficient_Type"] == coefficient_type]
        base_palette = {s: sample_colors[s][0] for s in active_subunits}
        coefficient_label = coefficient_display_label(coefficient_type)
        draw_coefficient_boxplot(
            ax=ax,
            panel_df=panel_df,
            x_col="Subunit",
            order=active_subunits,
            palette=base_palette,
            y_label="Coefficient",
            title=f"{coefficient_label} Coefficient Distribution",
            y_min=y_mins[coefficient_type],
            y_limit=y_limits[coefficient_type],
        )

    plt.tight_layout()
    plt.savefig(
        build_graph_path(output_root, "Joint Subunit", f"{filename_prefix}_boxplot_separate.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    for coefficient_type in coefficient_types:
        fig, ax = plt.subplots(figsize=(8, 6))
        coefficient_label = coefficient_display_label(coefficient_type)
        draw_coefficient_boxplot(
            ax=ax,
            panel_df=box_df[box_df["Coefficient_Type"] == coefficient_type],
            x_col="Subunit",
            order=active_subunits,
            palette={s: sample_colors[s][0] for s in active_subunits},
            y_label=f"{coefficient_label} Coefficient",
            title=f"{coefficient_label} Coefficient Distribution",
            y_min=y_mins[coefficient_type],
            y_limit=y_limits[coefficient_type],
        )
        plt.tight_layout()
        plt.savefig(
            build_graph_path(
                output_root,
                "Joint Subunit",
                f"{filename_prefix}_boxplot_{coefficient_slug(coefficient_type)}_only.png",
            ),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    if not replicate_order:
        return

    fig, axes = create_subunit_grid(len(active_subunits), sharex=False, sharey=True)
    for ax, subunit in zip(axes, active_subunits):
        first_type = coefficient_types[0]
        subunit_first_df = box_df[
            (box_df["Subunit"] == subunit) & (box_df["Coefficient_Type"] == first_type)
        ]
        replicate_palette = build_replicate_palette(subunit, replicate_order)
        draw_coefficient_boxplot(
            ax=ax,
            panel_df=subunit_first_df,
            x_col="Replicate_Label",
            order=replicate_order,
            palette=replicate_palette,
            y_label=f"{coefficient_display_label(first_type)} Coefficient",
            title=subunit,
            y_min=y_mins[first_type],
            y_limit=y_limits[first_type],
        )
        ax.set_xlabel("Replicate", fontsize=28, fontweight="bold", color="grey")
        ax.tick_params(axis="x", labelbottom=True)
        if subunit != active_subunits[0]:
            ax.set_ylabel("")
    hide_unused_axes(axes, len(active_subunits))
    plt.tight_layout()
    plt.savefig(
        build_graph_path(
            output_root,
            "Joint Subunit",
            f"replicate_{coefficient_slug(first_type)}_boxplot_grid.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    fig, axes = plt.subplots(
        len(active_subunits),
        len(coefficient_types),
        figsize=(6 * len(coefficient_types), max(4.5 * len(active_subunits), 6)),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    for row_idx, subunit in enumerate(active_subunits):
        for col_idx, coefficient_type in enumerate(coefficient_types):
            ax = axes[row_idx, col_idx]
            panel_df = box_df[
                (box_df["Subunit"] == subunit) & (box_df["Coefficient_Type"] == coefficient_type)
            ]
            coefficient_label = coefficient_display_label(coefficient_type)
            replicate_palette = build_replicate_palette(subunit, replicate_order)
            draw_coefficient_boxplot(
                ax=ax,
                panel_df=panel_df,
                x_col="Replicate_Label",
                order=replicate_order,
                palette=replicate_palette,
                y_label=f"{coefficient_label} Coefficient",
                title=f"{subunit} {coefficient_label}",
                y_min=y_mins[coefficient_type],
                y_limit=y_limits[coefficient_type],
            )
            ax.set_xlabel("Replicate", fontsize=28, fontweight="bold", color="grey")
            ax.tick_params(axis="x", labelbottom=True)
    plt.tight_layout()
    plt.savefig(
        build_graph_path(output_root, "Joint Subunit", f"replicate_{filename_prefix}_boxplot_grid.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    for subunit in active_subunits:
        for coefficient_type in coefficient_types:
            panel_df = box_df[
                (box_df["Subunit"] == subunit) & (box_df["Coefficient_Type"] == coefficient_type)
            ]
            if panel_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            coefficient_label = coefficient_display_label(coefficient_type)
            replicate_palette = build_replicate_palette(subunit, replicate_order)
            draw_coefficient_boxplot(
                ax=ax,
                panel_df=panel_df,
                x_col="Replicate_Label",
                order=replicate_order,
                palette=replicate_palette,
                y_label=f"{coefficient_label} Coefficient",
                title=f"{subunit} {coefficient_label} by Replicate",
                y_min=y_mins[coefficient_type],
                y_limit=y_limits[coefficient_type],
            )
            ax.set_xlabel("Replicate", fontsize=28, fontweight="bold", color="grey")
            plt.tight_layout()
            plt.savefig(
                build_graph_path(
                    output_root,
                    subunit,
                    f"{subunit}_{coefficient_slug(coefficient_type)}_replicate_boxplot.png",
                ),
                dpi=600,
                bbox_inches="tight",
            )
            plt.close()


def plot_violinplots(coefficient_df, output_root, coefficient_types, y_mins, y_limits, filename_prefix):
    if coefficient_df.empty:
        return

    violin_df = coefficient_df.copy()
    active_subunits = get_present_subunits(violin_df)
    if not active_subunits:
        return

    violin_df["Subunit"] = pd.Categorical(violin_df["Subunit"], categories=active_subunits, ordered=True)
    violin_df["Coefficient_Type"] = pd.Categorical(
        violin_df["Coefficient_Type"],
        categories=coefficient_types,
        ordered=True,
    )
    replicate_order = get_replicate_order(violin_df)
    if replicate_order:
        violin_df["Replicate_Label"] = pd.Categorical(
            violin_df["Replicate_Label"],
            categories=replicate_order,
            ordered=True,
        )
    violin_df = violin_df.sort_values(["Subunit", "Coefficient_Type"])

    fig, axes = plt.subplots(1, len(coefficient_types), figsize=(7 * len(coefficient_types), 6), sharey=False)
    if len(coefficient_types) == 1:
        axes = [axes]

    for ax, coefficient_type in zip(axes, coefficient_types):
        panel_df = violin_df[violin_df["Coefficient_Type"] == coefficient_type]
        coefficient_label = coefficient_display_label(coefficient_type)
        draw_coefficient_violinplot(
            ax=ax,
            panel_df=panel_df,
            x_col="Subunit",
            order=active_subunits,
            palette={s: sample_colors[s][0] for s in active_subunits},
            y_label="Coefficient",
            title=f"{coefficient_label} Coefficient Distribution",
            y_min=y_mins[coefficient_type],
            y_limit=y_limits[coefficient_type],
        )

    plt.tight_layout()
    plt.savefig(
        build_graph_path(output_root, "Joint Subunit", f"{filename_prefix}_violinplot_separate.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    for coefficient_type in coefficient_types:
        fig, ax = plt.subplots(figsize=(8, 6))
        coefficient_label = coefficient_display_label(coefficient_type)
        draw_coefficient_violinplot(
            ax=ax,
            panel_df=violin_df[violin_df["Coefficient_Type"] == coefficient_type],
            x_col="Subunit",
            order=active_subunits,
            palette={s: sample_colors[s][0] for s in active_subunits},
            y_label=f"{coefficient_label} Coefficient",
            title=f"{coefficient_label} Coefficient Distribution",
            y_min=y_mins[coefficient_type],
            y_limit=y_limits[coefficient_type],
        )
        plt.tight_layout()
        plt.savefig(
            build_graph_path(
                output_root,
                "Joint Subunit",
                f"{filename_prefix}_violinplot_{coefficient_slug(coefficient_type)}_only.png",
            ),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    if not replicate_order:
        return

    fig, axes = create_subunit_grid(len(active_subunits), sharex=False, sharey=True)
    for ax, subunit in zip(axes, active_subunits):
        first_type = coefficient_types[0]
        subunit_first_df = violin_df[
            (violin_df["Subunit"] == subunit) & (violin_df["Coefficient_Type"] == first_type)
        ]
        draw_coefficient_violinplot(
            ax=ax,
            panel_df=subunit_first_df,
            x_col="Replicate_Label",
            order=replicate_order,
            palette=build_replicate_palette(subunit, replicate_order),
            y_label=f"{coefficient_display_label(first_type)} Coefficient",
            title=subunit,
            y_min=y_mins[first_type],
            y_limit=y_limits[first_type],
        )
        ax.set_xlabel("Replicate", fontsize=28, fontweight="bold", color="grey")
        ax.tick_params(axis="x", labelbottom=True)
        if subunit != active_subunits[0]:
            ax.set_ylabel("")
    hide_unused_axes(axes, len(active_subunits))
    plt.tight_layout()
    plt.savefig(
        build_graph_path(
            output_root,
            "Joint Subunit",
            f"replicate_{coefficient_slug(first_type)}_violinplot_grid.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    fig, axes = plt.subplots(
        len(active_subunits),
        len(coefficient_types),
        figsize=(6 * len(coefficient_types), max(4.5 * len(active_subunits), 6)),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    for row_idx, subunit in enumerate(active_subunits):
        for col_idx, coefficient_type in enumerate(coefficient_types):
            ax = axes[row_idx, col_idx]
            panel_df = violin_df[
                (violin_df["Subunit"] == subunit) & (violin_df["Coefficient_Type"] == coefficient_type)
            ]
            coefficient_label = coefficient_display_label(coefficient_type)
            draw_coefficient_violinplot(
                ax=ax,
                panel_df=panel_df,
                x_col="Replicate_Label",
                order=replicate_order,
                palette=build_replicate_palette(subunit, replicate_order),
                y_label=f"{coefficient_label} Coefficient",
                title=f"{subunit} {coefficient_label}",
                y_min=y_mins[coefficient_type],
                y_limit=y_limits[coefficient_type],
            )
            ax.set_xlabel("Replicate", fontsize=28, fontweight="bold", color="grey")
            ax.tick_params(axis="x", labelbottom=True)
    plt.tight_layout()
    plt.savefig(
        build_graph_path(output_root, "Joint Subunit", f"replicate_{filename_prefix}_violinplot_grid.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    for subunit in active_subunits:
        for coefficient_type in coefficient_types:
            panel_df = violin_df[
                (violin_df["Subunit"] == subunit) & (violin_df["Coefficient_Type"] == coefficient_type)
            ]
            if panel_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            coefficient_label = coefficient_display_label(coefficient_type)
            draw_coefficient_violinplot(
                ax=ax,
                panel_df=panel_df,
                x_col="Replicate_Label",
                order=replicate_order,
                palette=build_replicate_palette(subunit, replicate_order),
                y_label=f"{coefficient_label} Coefficient",
                title=f"{subunit} {coefficient_label} by Replicate",
                y_min=y_mins[coefficient_type],
                y_limit=y_limits[coefficient_type],
            )
            ax.set_xlabel("Replicate", fontsize=28, fontweight="bold", color="grey")
            plt.tight_layout()
            plt.savefig(
                build_graph_path(
                    output_root,
                    subunit,
                    f"{subunit}_{coefficient_slug(coefficient_type)}_replicate_violinplot.png",
                ),
                dpi=600,
                bbox_inches="tight",
            )
            plt.close()


# ==========================================================
# LOAD + COPY INPUT CSVS
# ==========================================================

source_files = discover_candidate_csvs(input_folder)
if not source_files:
    raise ValueError("No matching biphasic CSV files were found.")

all_dfs = []

for path in source_files:
    metadata = detect_metadata(path)
    subunit_folder = os.path.join(input_copy_folder, sanitize_name(metadata["Subunit"]))
    os.makedirs(subunit_folder, exist_ok=True)
    destination = os.path.join(subunit_folder, os.path.basename(path))
    shutil.copy(path, destination)
    print(f"Copied {os.path.basename(path)} -> {subunit_folder}")

    loaded = load_biphasic_components(path)
    all_dfs.append(loaded)
    print(f"Loaded: {os.path.basename(path)}")

raw_df = pd.concat(all_dfs, ignore_index=True)
raw_df = assign_replicate_labels(raw_df)
raw_df.to_csv(os.path.join(csv_folder, "all_detected_biphasic_components.csv"), index=False)


# ==========================================================
# BUILD ANALYSIS TABLES
# ==========================================================

component_df, coefficient_df, dwell_coefficient_df = build_component_and_coefficient_tables(raw_df)

if component_df.empty:
    raise ValueError("No complete biphasic up/down event pairs were found.")

component_df.to_csv(os.path.join(csv_folder, "biphasic_component_events.csv"), index=False)
coefficient_df.to_csv(os.path.join(csv_folder, "biphasic_coefficients.csv"), index=False)
dwell_coefficient_df.to_csv(os.path.join(csv_folder, "biphasic_dwell_coefficients.csv"), index=False)

component_plot_df = clean_plot_df(
    component_df,
    x_col="Dwell Time (ms)",
    y_col="Amplitude (pA)",
    x_max=4,
    y_max=600,
)

coefficient_plot_df = clean_plot_df(
    coefficient_df,
    x_col="Dwell Time (ms)",
    y_col="Coefficient",
    x_max=3,
    y_max=10,
)

dwell_coefficient_plot_df = clean_plot_df(
    dwell_coefficient_df,
    x_col="Dwell Time (ms)",
    y_col="Coefficient",
    x_max=3,
    y_max=10,
)

if component_plot_df.empty:
    raise ValueError("No component events remained after plotting filters.")

if coefficient_plot_df.empty:
    raise ValueError("No coefficient values remained after plotting filters.")

if dwell_coefficient_plot_df.empty:
    raise ValueError("No dwell coefficient values remained after plotting filters.")


# ==========================================================
# PLOTS
# ==========================================================

amplitude_bins = y_bins_from_limit(600, AMPLITUDE_BIN_WIDTH)
coefficient_bins = y_bins_from_limit(10, COEFFICIENT_BIN_WIDTH)
ic_bins = y_bins_from_limit(IC_Y_MAX, COEFFICIENT_BIN_WIDTH, y_min=IC_Y_MIN)

for subunit in subunit_order:
    subunit_component_df = component_plot_df[component_plot_df["Subunit"] == subunit]
    subunit_coefficient_df = coefficient_plot_df[coefficient_plot_df["Subunit"] == subunit]
    subunit_dwell_coefficient_df = dwell_coefficient_plot_df[dwell_coefficient_plot_df["Subunit"] == subunit]

    if not subunit_component_df.empty:
        plot_joint_component(
            subunit_df=subunit_component_df,
            output_path=build_graph_path(amplitude_graph_folder, subunit, f"{subunit}_dwell_vs_amplitude_up_down_joint.png"),
            y_label="Amplitude (pA)",
            y_col="Amplitude (pA)",
            x_col="Dwell Time (ms)",
            title_suffix="Up and Down Biphasic Components",
            x_limit=4,
            y_min=0,
            y_limit=600,
            y_bins=amplitude_bins,
            up_label="I$_r$",
            down_label="I$_c$",
        )

    ic_df = subunit_coefficient_df[subunit_coefficient_df["Coefficient_Type"] == "Ic"]
    if GENERATE_IC_IR_JOINT_SCATTERS and (not ic_df.empty):
        plot_joint_component(
            subunit_df=ic_df,
            output_path=build_graph_path(coefficient_graph_folder, subunit, f"{subunit}_dwell_vs_coefficient_ic_joint.png"),
            y_label="I$_c$ Coefficient",
            y_col="Coefficient",
            x_col="Dwell Time (ms)",
            title_suffix="I$_c$ Coefficient",
            x_limit=3,
            y_min=IC_Y_MIN,
            y_limit=IC_Y_MAX,
            y_bins=ic_bins,
            use_direction_shading=False,
            color=sample_colors.get(subunit, ("#95a5a6", "#34495e"))[0],
            legend_label="I$_c$",
            show_legend=False,
        )

    ir_df = subunit_coefficient_df[subunit_coefficient_df["Coefficient_Type"] == "Ir"]
    if GENERATE_IC_IR_JOINT_SCATTERS and (not ir_df.empty):
        plot_joint_component(
            subunit_df=ir_df,
            output_path=build_graph_path(coefficient_graph_folder, subunit, f"{subunit}_dwell_vs_coefficient_ir_joint.png"),
            y_label="I$_r$ Coefficient",
            y_col="Coefficient",
            x_col="Dwell Time (ms)",
            title_suffix="I$_r$ Coefficient",
            x_limit=3,
            y_min=0,
            y_limit=10,
            y_bins=coefficient_bins,
            use_direction_shading=False,
            color=sample_colors.get(subunit, ("#95a5a6", "#34495e"))[0],
            legend_label="I$_r$",
            show_legend=False,
        )

    dc_df = subunit_dwell_coefficient_df[subunit_dwell_coefficient_df["Coefficient_Type"] == "Dc"]
    if not dc_df.empty:
        plot_joint_component(
            subunit_df=dc_df,
            output_path=build_graph_path(dwell_time_graph_folder, subunit, f"{subunit}_dwell_vs_coefficient_dc_joint.png"),
            y_label="D$_c$ Coefficient",
            y_col="Coefficient",
            x_col="Dwell Time (ms)",
            title_suffix="D$_c$ Coefficient",
            x_limit=3,
            y_min=IC_Y_MIN,
            y_limit=IC_Y_MAX,
            y_bins=ic_bins,
            use_direction_shading=False,
            color=sample_colors.get(subunit, ("#95a5a6", "#34495e"))[0],
            legend_label="D$_c$",
            show_legend=False,
        )

    dr_df = subunit_dwell_coefficient_df[subunit_dwell_coefficient_df["Coefficient_Type"] == "Dr"]
    if not dr_df.empty:
        plot_joint_component(
            subunit_df=dr_df,
            output_path=build_graph_path(dwell_time_graph_folder, subunit, f"{subunit}_dwell_vs_coefficient_dr_joint.png"),
            y_label="D$_r$ Coefficient",
            y_col="Coefficient",
            x_col="Dwell Time (ms)",
            title_suffix="D$_r$ Coefficient",
            x_limit=3,
            y_min=0,
            y_limit=10,
            y_bins=coefficient_bins,
            use_direction_shading=False,
            color=sample_colors.get(subunit, ("#95a5a6", "#34495e"))[0],
            legend_label="D$_r$",
            show_legend=False,
        )

plot_three_panel_grid(
    plot_df=component_plot_df,
    output_path=build_graph_path(amplitude_graph_folder, "Joint Subunit", "40S_60S_80S_dwell_vs_amplitude_up_down_grid.png"),
    y_label="Amplitude (pA)",
    y_col="Amplitude (pA)",
    x_col="Dwell Time (ms)",
    x_limit=4,
    y_min=0,
    y_limit=600,
    y_bins=amplitude_bins,
    title="Biphasic Up and Down Components",
    use_direction_shading=True,
    show_subplot_titles=False,
    panel_label_side="right",
    up_label="I$_r$",
    down_label="I$_c$",
)

plot_three_panel_grid(
    plot_df=coefficient_plot_df[coefficient_plot_df["Coefficient_Type"] == "Ic"],
    output_path=build_graph_path(coefficient_graph_folder, "Joint Subunit", "40S_60S_80S_dwell_vs_coefficient_ic_grid.png"),
    y_label="I$_c$ Coefficient",
    y_col="Coefficient",
    x_col="Dwell Time (ms)",
    x_limit=3,
    y_min=IC_Y_MIN,
    y_limit=IC_Y_MAX,
    y_bins=ic_bins,
    title="Biphasic I$_c$ Coefficient",
    use_direction_shading=False,
    legend_label="I$_c$",
    show_legend=False,
    show_subplot_titles=False,
    panel_label_side="right",
)

plot_three_panel_grid(
    plot_df=coefficient_plot_df[coefficient_plot_df["Coefficient_Type"] == "Ir"],
    output_path=build_graph_path(coefficient_graph_folder, "Joint Subunit", "40S_60S_80S_dwell_vs_coefficient_ir_grid.png"),
    y_label="I$_r$ Coefficient",
    y_col="Coefficient",
    x_col="Dwell Time (ms)",
    x_limit=3,
    y_min=0,
    y_limit=10,
    y_bins=coefficient_bins,
    title="Biphasic I$_r$ Coefficient",
    use_direction_shading=False,
    legend_label="I$_r$",
    show_legend=False,
    show_subplot_titles=False,
    panel_label_side="right",
)

plot_combined_coefficient_joint_grid(
    plot_df=coefficient_plot_df,
    output_path=build_graph_path(coefficient_graph_folder, "Joint Subunit", "40S_60S_80S_dwell_vs_coefficient_ic_ir_grid.png"),
    x_limit=3,
    y_min=0,
    y_limit=10,
    y_bins=coefficient_bins,
    title="Biphasic I$_c$ and I$_r$ Coefficients",
    coefficient_types=["Ir", "Ic"],
)

plot_combined_coefficient_joint_grid(
    plot_df=dwell_coefficient_plot_df,
    output_path=build_graph_path(dwell_time_graph_folder, "Joint Subunit", "40S_60S_80S_dwell_vs_coefficient_dc_dr_grid.png"),
    x_limit=3,
    y_min=0,
    y_limit=10,
    y_bins=coefficient_bins,
    title="Biphasic D$_c$ and D$_r$ Coefficients",
    coefficient_types=["Dr", "Dc"],
)

if GENERATE_IC_IR_JOINT_SCATTERS:
    plot_overlay_joint_by_subunit(
        plot_df=coefficient_plot_df,
        output_path=build_graph_path(coefficient_graph_folder, "Joint Subunit", "all_subunits_dwell_vs_coefficient_ic_joint.png"),
        coefficient_type="Ic",
        title_suffix="All Subunits I$_c$ Coefficient",
        x_limit=3,
        y_min=IC_Y_MIN,
        y_limit=IC_Y_MAX,
        y_bins=ic_bins,
    )

    plot_overlay_joint_by_subunit(
        plot_df=coefficient_plot_df,
        output_path=build_graph_path(coefficient_graph_folder, "Joint Subunit", "all_subunits_dwell_vs_coefficient_ir_joint.png"),
        coefficient_type="Ir",
        title_suffix="All Subunits I$_r$ Coefficient",
        x_limit=3,
        y_min=0,
        y_limit=10,
        y_bins=coefficient_bins,
    )

plot_overlay_coefficient_grid(
    plot_df=coefficient_plot_df,
    output_path=build_graph_path(coefficient_graph_folder, "Joint Subunit", "all_subunits_dwell_vs_coefficient_ic_ir_grid.png"),
    x_limit=3,
    y_mins={"Ic": IC_Y_MIN, "Ir": 0},
    y_limits={"Ic": IC_Y_MAX, "Ir": 10},
    y_bins_map={"Ic": ic_bins, "Ir": coefficient_bins},
    title="All Subunits I$_c$ and I$_r$ Coefficients",
    coefficient_types=["Ic", "Ir"],
)

plot_overlay_coefficient_grid(
    plot_df=dwell_coefficient_plot_df,
    output_path=build_graph_path(dwell_time_graph_folder, "Joint Subunit", "all_subunits_dwell_vs_coefficient_dc_dr_grid.png"),
    x_limit=3,
    y_mins={"Dc": IC_Y_MIN, "Dr": 0},
    y_limits={"Dc": IC_Y_MAX, "Dr": 10},
    y_bins_map={"Dc": ic_bins, "Dr": coefficient_bins},
    title="All Subunits D$_c$ and D$_r$ Coefficients",
    coefficient_types=["Dc", "Dr"],
)

plot_three_panel_grid(
    plot_df=dwell_coefficient_plot_df[dwell_coefficient_plot_df["Coefficient_Type"] == "Dc"],
    output_path=build_graph_path(dwell_time_graph_folder, "Joint Subunit", "40S_60S_80S_dwell_vs_coefficient_dc_grid.png"),
    y_label="D$_c$ Coefficient",
    y_col="Coefficient",
    x_col="Dwell Time (ms)",
    x_limit=3,
    y_min=IC_Y_MIN,
    y_limit=IC_Y_MAX,
    y_bins=ic_bins,
    title="Biphasic D$_c$ Coefficient",
    use_direction_shading=False,
    legend_label="D$_c$",
    show_legend=False,
    show_subplot_titles=False,
    panel_label_side="right",
)

plot_three_panel_grid(
    plot_df=dwell_coefficient_plot_df[dwell_coefficient_plot_df["Coefficient_Type"] == "Dr"],
    output_path=build_graph_path(dwell_time_graph_folder, "Joint Subunit", "40S_60S_80S_dwell_vs_coefficient_dr_grid.png"),
    y_label="D$_r$ Coefficient",
    y_col="Coefficient",
    x_col="Dwell Time (ms)",
    x_limit=3,
    y_min=0,
    y_limit=10,
    y_bins=coefficient_bins,
    title="Biphasic D$_r$ Coefficient",
    use_direction_shading=False,
    legend_label="D$_r$",
    show_legend=False,
    show_subplot_titles=False,
    panel_label_side="right",
)

plot_overlay_joint_by_subunit(
    plot_df=dwell_coefficient_plot_df,
    output_path=build_graph_path(dwell_time_graph_folder, "Joint Subunit", "all_subunits_dwell_vs_coefficient_dc_joint.png"),
    coefficient_type="Dc",
    title_suffix="All Subunits D$_c$ Coefficient",
    x_limit=3,
    y_min=IC_Y_MIN,
    y_limit=IC_Y_MAX,
    y_bins=ic_bins,
)

plot_overlay_joint_by_subunit(
    plot_df=dwell_coefficient_plot_df,
    output_path=build_graph_path(dwell_time_graph_folder, "Joint Subunit", "all_subunits_dwell_vs_coefficient_dr_joint.png"),
    coefficient_type="Dr",
    title_suffix="All Subunits D$_r$ Coefficient",
    x_limit=3,
    y_min=0,
    y_limit=10,
    y_bins=coefficient_bins,
)

plot_boxplots(
    coefficient_plot_df,
    output_root=coefficient_graph_folder,
    coefficient_types=["Ic", "Ir"],
    y_mins={"Ic": IC_Y_MIN, "Ir": 0},
    y_limits={"Ic": IC_Y_MAX, "Ir": 10},
    filename_prefix="coefficient",
    summary_filename="biphasic_coefficient_summary.csv",
)

plot_violinplots(
    coefficient_plot_df,
    output_root=coefficient_graph_folder,
    coefficient_types=["Ic", "Ir"],
    y_mins={"Ic": IC_Y_MIN, "Ir": 0},
    y_limits={"Ic": IC_Y_MAX, "Ir": 10},
    filename_prefix="coefficient",
)

plot_boxplots(
    dwell_coefficient_plot_df,
    output_root=dwell_time_graph_folder,
    coefficient_types=["Dc", "Dr"],
    y_mins={"Dc": IC_Y_MIN, "Dr": 0},
    y_limits={"Dc": IC_Y_MAX, "Dr": 10},
    filename_prefix="dwell_coefficient",
    summary_filename="biphasic_dwell_coefficient_summary.csv",
)

plot_violinplots(
    dwell_coefficient_plot_df,
    output_root=dwell_time_graph_folder,
    coefficient_types=["Dc", "Dr"],
    y_mins={"Dc": IC_Y_MIN, "Dr": 0},
    y_limits={"Dc": IC_Y_MAX, "Dr": 10},
    filename_prefix="dwell_coefficient",
)

plot_coefficient_histograms(
    coefficient_plot_df,
    output_root=coefficient_graph_folder,
    coefficient_type="Ic",
    x_min=IC_Y_MIN,
    x_max=IC_Y_MAX,
    bin_width=IC_HISTOGRAM_BIN_WIDTH,
)


# ==========================================================
# COPY SCRIPT
# ==========================================================

shutil.copy(os.path.realpath(__file__), os.path.join(output_folder, os.path.basename(__file__)))


# ==========================================================
# OUTPUT TEXT
# ==========================================================

print("\n✅ Biphasic summary pipeline complete.")
print(output_folder)

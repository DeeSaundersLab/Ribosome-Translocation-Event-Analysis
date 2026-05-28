# Ribosome Translocation Event Analysis Pipeline

A Python pipeline for analysing nanopore ribosome translocation experiments from exported CSV event files. The script automatically concatenates replicates, detects monophasic and biphasic events, performs data cleaning, extracts experiment metadata from folder structures, and generates publication-quality visualisations for dwell time and amplitude analysis.

Designed for single-molecule nanopore electrophysiology workflows involving ribosomal subunits and translocation event characterisation.

---

# Features

* Automatic metadata detection from folder names:

  * Ribosomal subunits
  * Concentration
  * Applied voltage
* Concatenates multiple replicate CSV files
* Cleans and standardises event datasets
* Detects:

  * Monophasic events
  * Biphasic translocation events
* Extracts event coordinates and timing
* Generates:

  * Joint scatter/histogram plots
  * Grid comparison plots
  * Monophasic vs biphasic comparisons
* Produces publication-quality figures at 600 DPI
* Automatically organises outputs into timestamped folders
* Copies original input CSVs and analysis script for reproducibility

---

# Applications

Useful for:

* Nanopore ribosome translocation analysis
* Single-molecule electrophysiology
* Event classification workflows
* Dwell time and blockade amplitude analysis
* Quartz nanopipette nanopore experiments
* Ribosomal subunit comparison studies

---

# Dependencies

Install required packages using:

```bash
pip install pandas numpy matplotlib seaborn
```

---

# Input Requirements

The pipeline expects CSV event files exported from nanopore event detection software.

Required columns include:

```text
Coordinates
Direction
Amplitude (pA)
Duration (ms)
Area (pC)
```

---

# Folder Structure

The script extracts metadata directly from the folder hierarchy.

Example:

```text
80S/
├── 3.25ng/
│   ├── -700mV/
│   │   ├── baseline -1650/
│   │   │   ├── rep1.csv
│   │   │   ├── rep2.csv
```

Detected metadata:

* Sample: 80S
* Concentration: 3.25 ng
* Voltage: -700 mV

---

# Biphasic Event Detection

The pipeline identifies biphasic events by detecting sequential:

```text
Up event → Down event
```

with overlapping or adjacent coordinates.

Merged biphasic events include:

* Combined amplitude
* Combined duration
* Combined event area
* Shared event IDs

Both processed and unprocessed biphasic datasets are exported.

---

# Generated Outputs

The script automatically creates:

```text
Processed_TIMESTAMP/
├── CSV/
│   ├── concatenated_input.csv
│   ├── monophasic.csv
│   ├── biphasic.csv
│   ├── unprocessed_biphasic.csv
│
├── Graphs/
│   ├── all_events.png
│   ├── monophasic_events.png
│   ├── biphasic_events.png
│   ├── grid_events.png
│
├── Input_CSVs/
```

---

# Visualisations

Generated plots include:

* Dwell time vs amplitude scatter plots
* Marginal histograms
* Event-type comparisons
* Replicate-combined datasets
* Publication-ready figures with consistent axis scaling

---

# Usage

1. Set the input folder path:

```python
input_folder = "path/to/experiment/folder"
```

2. Run the script:

```bash
python ribosome_translocation_pipeline.py
```

3. Processed data and figures will be generated automatically.

---

# Output Filtering

Plots are filtered to:

* Duration: 0–6 ms
* Amplitude: 0–900 pA

This improves visual consistency for nanopore translocation event analysis.

---

# Notes

Developed for nanopore-based ribosome translocation experiments using quartz nanopipettes and solid-state nanopore electrophysiology workflows.

# Nanopore Electrophysiology & Ribosome Translocation Analysis Toolkit

A collection of Python pipelines for analysing nanopore electrophysiology experiments, ribosome translocation events, ionic current traces, and nanopore conductance measurements.

The repository automates workflows from raw `.abf` recordings to publication-quality figures, event classification, I–V analysis, and nanopore size estimation.

Designed for quartz nanopipette and solid-state nanopore experiments involving ribosomal subunits, polymer electrolytes, and single-molecule translocation studies.

---

# Repository Contents

## 1. Ribosome Translocation Event Analysis Pipeline

Processes nanopore event CSV files to detect and classify:

* Monophasic events
* Biphasic events
* Ribosome translocation signatures

### Features

* Automatic metadata detection from folder structure
* Replicate concatenation
* Event cleaning and parsing
* Biphasic event merging
* Dwell time vs amplitude analysis
* Publication-quality scatter/histogram plots
* Event classification exports

### Outputs

* Processed CSV datasets
* Monophasic/biphasic event tables
* Joint plots
* Grid comparison figures

---

## 2. I–V Curve & Nanopore Size Estimator

Processes electrophysiology `.abf` files to generate:

* I–V curves
* Linear regression fits
* Conductance measurements
* Nanopore resistance calculations
* Nanopore diameter estimations

### Features

* ABF → CSV conversion
* Electrolyte auto-detection
* Mean ± SEM analysis
* Combined replicate plotting
* Nanopore geometry estimation
* Publication-quality I–V figures

### Supported Electrolytes

* 0.1 M KCl
* 3 M KCl
* PEG + KCl systems

### Outputs

* I–V CSV datasets
* Regression summaries
* Conductance plots
* Nanopore size estimation tables

---

## 3. Ionic Current Plotting Pipeline

Visualises ionic current traces directly from `.abf` files.

### Features

* Automatic sample detection from file paths
* Global y-axis normalisation
* Individual trace plotting
* Combined trace plotting
* Consistent colour coding for ribosomal subunits
* High-resolution PNG export

### Outputs

* Single-trace current plots
* Combined ionic current overlays
* Timestamped figure folders

---

# Applications

Useful for:

* Nanopore electrophysiology
* Quartz nanopipette analysis
* Ribosome translocation studies
* Single-molecule sensing
* Solid-state nanopore research
* Conductance and resistance measurements
* Polymer electrolyte nanopore systems

---

# Dependencies

Install required packages:

```bash
pip install numpy pandas matplotlib seaborn pyabf
```

---

# Supported File Types

| File Type | Purpose                                |
| --------- | -------------------------------------- |
| `.abf`    | Electrophysiology recordings           |
| `.csv`    | Event detection and processed datasets |
| `.png`    | Publication-quality exported figures   |

---

# Example Workflow

```text
Raw ABF Files
      ↓
Ionic Current Plotting
      ↓
Event Detection CSVs
      ↓
Translocation Event Analysis
      ↓
I–V Analysis & Nanopore Estimation
      ↓
Publication Figures & Processed Data
```

---

# Generated Outputs

The toolkit automatically creates timestamped output folders containing:

```text
Processed_TIMESTAMP/
├── CSV/
├── Graphs/
├── Plots/
├── Linear_Regression/
├── Input_CSVs/
```

---

# Visualisations

Generated figures include:

* Ionic current traces
* Dwell time vs amplitude plots
* Marginal histograms
* Monophasic vs biphasic comparisons
* I–V curves
* Linear regression fits
* Mean ± SEM conductance plots
* Combined replicate overlays

All figures are exported as high-resolution PNGs suitable for publication and presentations.

---

# Usage

1. Set the input folder path within the script
2. Run the desired pipeline:

```bash
python script_name.py
```

3. Processed data and figures will be generated automatically.

---

# Notes

Developed for nanopore-based ribosome translocation and electrophysiology workflows using quartz nanopipettes and Axon Instruments recording systems.

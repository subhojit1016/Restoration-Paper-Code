# Restoration-Paper-Code

Python implementations and input datasets for numerical experiments on network restoration routing, prepared as supporting research materials for an accepted paper.

**Publication information:** Paper title, author list, publication venue, and DOI will be added when finalized. Until then, please cite the associated paper rather than treating this repository as a separate publication.

## Contents

The current repository contains **14 Python scripts**, **60 network-instance text files**, and **16 realization CSV files**.

```text
Restoration-Paper-Code/
├── src/
│   ├── algorithms/           # Restoration algorithms and heuristic comparisons
│   ├── instance_generation/  # Network-instance generation
│   ├── analysis/             # Result aggregation and table preparation
│   └── utilities/            # Supporting data and plotting utilities
├── data/
│   ├── networks/             # Network input instances (.txt)
│   └── realizations/         # Stochastic realization inputs (.csv)
├── docs/
│   ├── FILE_INVENTORY.csv    # Inventory from repository preparation
│   └── INPUT_AUDIT.md       # Preliminary input-reference audit
├── requirements.txt
├── .gitignore
└── README.md
```

## Python programs

| Location | Purpose |
|---|---|
| `src/algorithms/` | Restoration procedures, state-aggregation variants, and priority/nearest-neighbor heuristics |
| `src/instance_generation/` | Generate network instances used in experiments |
| `src/analysis/` | Combine and summarize experiment outputs for tables |
| `src/utilities/` | Auxiliary file processing, network checks, and illustrations |

The script filenames retain their original experiment labels. In particular, a number appearing in a script filename is **not** a guarantee that a matching input instance is included in this archive.

## Input data

- **Network instances:** `data/networks/` contains text files for supplied 20-, 30-, 40-, and 50-node network configurations, including files identified as cyclic, synthetic, or method-specific variants.
- **Stochastic realizations:** `data/realizations/` contains CSV files labeled by network size and parameter values `0.25`, `0.50`, `0.75`, and `0.90` for the 20-, 30-, 40-, and 50-node cases.
- The meaning of individual fields and parameters should be checked against the corresponding Python script and the accepted paper; filenames alone do not establish their statistical interpretation.

## Requirements

Use Python 3 and install the dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
# Activate the virtual environment for your operating system.
python -m pip install -r requirements.txt
```

Dependency versions are not pinned because the original software environment has not been confirmed.

## Running the experiments

The programs are individual research scripts rather than a unified command-line application. Before running a script:

1. Open the script and identify its expected input filenames, parameters, and output locations.
2. Locate the corresponding files in `data/networks/` and `data/realizations/`.
3. Check any relative paths or absolute paths in the script. Some scripts may expect inputs in their current working directory; moving files into the organized folders does not automatically update those paths.
4. Run the selected script from a working directory where its expected inputs can be found. For example:

```bash
python "src/algorithms/PP and NN.py"
```

**This command illustrates the invocation syntax only.** It has not been verified to run successfully from the repository root with the current input paths. Refer to `docs/INPUT_AUDIT.md` for known or suspected input gaps.

## Reproducibility status

The files are provided to document the research implementation and available experimental inputs. During repository preparation, comments were added to the Python scripts without intentionally changing their executable statements. However:

- The scripts have **not** been executed or systematically checked against the accepted paper's numerical tables and figures.
- Some hard-coded or relative input/output paths may require adjustment for a new environment.
- Some script-specific input instances or previously generated result files may not be present.
- No claim of one-command, end-to-end reproducibility is made at this stage.

For the exact inventory, consult `docs/FILE_INVENTORY.csv`. The input audit is preliminary and should not be treated as a completed runtime dependency check.

## Citation and licensing

If you use this code or the accompanying datasets, please cite the associated research paper. Full bibliographic details and a formal license should be added after confirmation by the authors and any applicable publisher requirements. **No license is implied by public availability on GitHub.**

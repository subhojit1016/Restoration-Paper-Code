# Restoration-Paper-Code

Research code and input instances associated with an accepted paper on network restoration routing. This repository combines the original Python source archive with the **updated** network-instance and realization files from `Codes for Github(1).zip`. Older input files are retained separately for traceability.

> **Publication note:** Add the final paper title, authors, journal/DOI, and citation once confirmed. The scripts have been **reorganized but not refactored or validated**; the results have not been reproduced as part of this packaging.

## Repository layout

| Directory | Contents |
|---|---|
| `src/algorithms/` | Restoration algorithm implementations, including SA, SM, PP and nearest-neighbor variants |
| `src/instance_generation/` | Instance and cyclic-network generation utilities |
| `src/analysis/` | Comparison and table-processing scripts |
| `src/utilities/` | Auxiliary checking and plotting scripts |
| `data/networks/` | Supplied network `.txt` instances |
| `data/realizations/` | Current realization `.csv` files (if supplied) |
| `data/previous_upload/` | Older or replaced inputs, preserved for reference only |
| `docs/` | File inventory and input-dependency audit |

## Installation

Requires Python 3 and the packages listed in `requirements.txt` (exact versions of the original environment are unknown).

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
```

## Running the original scripts

The scripts are preserved with their original names and content. Several refer to data files using relative paths, and some refer to files absent from the supplied archives. **Moving them into folders alone does not guarantee that they will run.**

For a script that uses relative input filenames, create a temporary working directory, copy the required network and realization inputs into it, and run the script from that directory (using the script's full path). Inspect the selected script for hard-coded paths, output destinations, and parameters before running it. Do not assume all scripts are runnable with the provided inputs.

See `docs/INPUT_AUDIT.md` for a preliminary filename-based audit and `docs/FILE_INVENTORY.csv` for the complete inventory.

## Reproducibility and limitations

- Python scripts are unchanged. Updated inputs are under `data/networks/` and `data/realizations/`; superseded and omitted earlier inputs are retained under `data/previous_upload/`.
- Do not mix archived files into current experiments without checking the intended version.
- Some referenced inputs and prior result files are missing.
- Code has not been executed or verified against the accepted paper's tables or figures.
- Dependencies are unpinned pending confirmation of the original software environment.
- Please check the publication's code/data policy and add licensing/citation metadata only after confirming the details.

## Citation

Please cite the accompanying accepted paper. Add its bibliographic details here when available.

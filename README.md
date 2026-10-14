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
├── jobs/
│   ├── run_experiment.slurm  # Generic HPC template
│   └── myjob_0.25_6_v2_SA1.slurm  # Author-provided experiment job
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

## Configuring experimental parameters

The numerical experiments depend on the probability parameter `p`, the service-time setting, the network instance, and other script-specific inputs. **Review the chosen Python script before running it.** Some programs contain settings that must be changed in the `.py` file; other programs accept positional arguments from a SLURM submission script. The latter should not be assumed for every program.

For the author-provided SLURM example, the following shell variables are passed to Python in this order:

| Shell variable | Example | How it is used |
|---|---|---|
| `x` | `0.25` | Intended probability setting (`p`); confirm in the target Python script |
| `y` | `6.0` | Intended service-time setting; confirm its units and meaning in the target script |
| `track` | `Network50R_4.txt` | Network-instance filename |
| `z` | `0.05` | Additional experiment parameter; see the target script |
| `i` | `1000000` | Additional integer setting; see the target script |

The command in the supplied job is:

```bash
python Algo_SA_0.25_Network50.py $x $y $track $z $i
```

To run another configuration, update the `x` and `y` values, select the appropriate network input, and review `z` and `i` as required. **These arguments affect the experiment only if the selected Python program reads them.** For programs that use hard-coded values instead, edit the relevant assignments in the `.py` file. Always verify that the selected network and realization inputs correspond to the intended experiment. Record parameter settings and outputs for each run.

## Running on an HPC cluster with SLURM

Two job examples are included in `jobs/`:

- [`myjob_0.25_6_v2_SA1.slurm`](jobs/myjob_0.25_6_v2_SA1.slurm) is the **author-provided job script**. It requests one task, 8 CPUs, 32 GB RAM, and a five-day wall-time limit. It loads `GCCcore/11.3.0` and `Python/3.10.4`, activates `Package/bin/activate`, and runs the Python command shown above. Its output pattern is `Example1Out.%j`.
- [`run_experiment.slurm`](jobs/run_experiment.slurm) is a separate, generic template with illustrative resource requests. It does not automatically pass the same parameters as the author-provided job.

The author-provided script uses the following relevant SLURM directives:

```bash
#SBATCH --job-name=JobExample1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=Example1Out.%j
#SBATCH --export=NONE
```

Before submitting a job, make sure the target `.py` file and `Network50R_4.txt` are available at the paths expected by the command. The script assumes its Python program can be invoked by filename from the job's working directory. Adjust the command, working directory, module names, and virtual-environment path for your cluster as needed. `--export=NONE` also means the environment is not simply inherited from the submitting shell; cluster-specific initialization may be necessary.

**Script selection and availability:** The exact file `Algo_SA_0.25_Network50.py` referenced in the example SLURM job is not included in this repository. However, the repository does include `src/algorithms/Algo_SA1_1stlevelaggregation_0.90_Network16.py` and `src/algorithms/Algo_SA2_2ndlevelaggregation_0.90_Network16.py`, which are SA1 and SA2 implementations labeled for `p=0.90` and 16-node networks. It also includes `src/algorithms/Algo_SA_3rdlevelaggregation_0.25_Network60.py`, labeled for SA3, `p=0.25`, and a 60-node network. These files should not be assumed interchangeable with the 50-node SA1 script named in the SLURM job. Before submitting a job, select the implementation and network size that match the intended experiment, confirm how the script receives `p`, service time, and other arguments, and update the Python command accordingly. The supplied SLURM job is a reference example and has not been validated against the included scripts.

Submit the job from a suitable working directory with:

```bash
sbatch jobs/myjob_0.25_6_v2_SA1.slurm
```

Check its status using `squeue -u "$USER"`. Review the `Example1Out.<jobid>` output file (and any cluster-specific error output) after execution. Resource requirements and software modules depend on the HPC system; the five-day wall time may not be permitted on every partition. Neither SLURM example has been executed as part of this repository preparation.

## Reproducibility status

The files are provided to document the research implementation and available experimental inputs. During repository preparation, comments were added to the Python scripts without intentionally changing their executable statements. However:

- The scripts have **not** been executed or systematically checked against the accepted paper's numerical tables and figures.
- Some hard-coded or relative input/output paths may require adjustment for a new environment.
- Some script-specific input instances or previously generated result files may not be present.
- No claim of one-command, end-to-end reproducibility is made at this stage.

For the exact inventory, consult `docs/FILE_INVENTORY.csv`. The input audit is preliminary and should not be treated as a completed runtime dependency check.

## Citation and licensing

If you use this code or the accompanying datasets, please cite the associated research paper. Full bibliographic details and a formal license should be added after confirmation by the authors and any applicable publisher requirements. **No license is implied by public availability on GitHub.**

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
│   └── run_SA1_example.slurm # Example SA1 HPC submission job
├── requirements.txt
├── .gitignore
└── README.md
```

## Python programs

The repository contains 14 Python scripts. The three state-aggregation implementations are named consistently:

| Method | Python script | Filename's example setting |
|---|---|---|
| SA1 | `src/algorithms/Algo_SA1_0.90_Network20.py` | `p = 0.90`, 20-node network |
| SA2 | `src/algorithms/Algo_SA2_0.90_Network20.py` | `p = 0.90`, 20-node network |
| SA3 | `src/algorithms/Algo_SA3_0.50_Network20.py` | `p = 0.50`, 20-node network |

Additional scripts in `src/algorithms/` implement other restoration procedures and heuristic comparisons. Scripts in `src/instance_generation/`, `src/analysis/`, and `src/utilities/` generate instances, aggregate results, and support the experiments.

**The parameter values in filenames are example labels, not universal settings.** To reproduce a particular case, check the parameter assignments or command-line arguments used by that script, along with the network and realization inputs. Do not assume that all three methods accept the same arguments without inspecting their code.

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

The experimental configuration depends on the probability parameter `p`, the service-time setting, the selected network, and any method-specific parameters. **Review and configure the selected `.py` script before running an experiment.** Some scripts accept command-line arguments; others may require updating parameter assignments or file paths inside the Python file. The original Python implementations are supplied unchanged.

For example, `Algo_SA1_0.90_Network20.py` reads five positional arguments:

| Argument | Example | Meaning in the SA1 script |
|---|---|---|
| `sys.argv[1]` | `0.90` | Probability parameter (`prob_value`) |
| `sys.argv[2]` | `6.0` | Service-time setting (`service`) |
| `sys.argv[3]` | `data/networks/Network20_Real_SA1.txt` | Network input filename |
| `sys.argv[4]` | `0.05` | `epsilon` |
| `sys.argv[5]` | `1000000` | `iteration` |

The supplied example job uses SA1. The SA2 and SA3 files are available in the same directory, but their argument handling and input assumptions should be checked separately before switching algorithms.

## Running an experiment using SLURM

An example batch job is provided at [`jobs/run_SA1_example.slurm`](jobs/run_SA1_example.slurm). It replaces the old job's reference to the absent `Algo_SA_0.25_Network50.py` with an **existing** repository script, `src/algorithms/Algo_SA1_0.90_Network20.py`, and uses the supplied `data/networks/Network20_Real_SA1.txt` instance.

The example requests one task, 8 CPUs, 32 GB of memory, and a five-day wall-time limit. It loads `GCCcore/11.3.0` and `Python/3.10.4` and activates `Package/bin/activate`, matching the author's supplied SLURM environment. These module and environment names are cluster-specific; create or select the appropriate environment and install the packages listed in `requirements.txt` before submitting the job.

Submit from the repository root:

```bash
sbatch jobs/run_SA1_example.slurm
```

Edit the job file to change `p`, `service_time`, `network`, `epsilon`, or `iterations` for a different experiment. Check that the network input is appropriate for the selected method and that the Python script supports the chosen configuration. The `1000000` iteration setting is taken from the supplied example and may be computationally expensive.

SLURM output and error logs use the patterns `restoration_SA1_%j.out` and `restoration_SA1_%j.err`, where `%j` is the SLURM job ID. The script assumes it is submitted from the repository root. Cluster-specific module initialization, permissions, environment setup, and paths may require changes.

**Reproducibility note:** The example is provided to document the submission procedure. It has not been execution-tested on the target cluster, and the presence of input files does not establish that every experiment reported in the paper can be reproduced without additional configuration.

#!/bin/bash
# Example SLURM submission for the SA1 restoration algorithm.
# Adjust resources, modules, environment, and parameters for your cluster.
#SBATCH --job-name=restoration_SA1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=restoration_SA1_%j.out
#SBATCH --error=restoration_SA1_%j.err
#SBATCH --export=NONE

set -e

# Submit this job from the repository root with:
#   sbatch jobs/run_SA1_example.slurm
# SLURM_SUBMIT_DIR is normally set to the submission directory.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

# Example HPC environment (adjust for your cluster).
module load GCCcore/11.3.0 Python/3.10.4
source Package/bin/activate

# Example settings; edit these for the intended experiment.
p=0.90                  # Probability parameter
service_time=6.0        # Service-time setting
epsilon=0.05            # Algorithm-specific parameter
iterations=1000000      # Algorithm-specific iteration count
network="data/networks/Network20_Real_SA1.txt"
script="src/algorithms/Algo_SA1_0.90_Network20.py"

# The SA1 Python script reads five positional arguments in this order:
# p, service_time, network_file, epsilon, iterations.
python "$script" "$p" "$service_time" "$network" "$epsilon" "$iterations"

#!/bin/bash
#SBATCH --account=zitnik_mz189              # associated Slurm account
#SBATCH --job-name=run_benchmark_%j           # assign job name
#SBATCH --ntasks-per-node=1                 # number of tasks per node
#SBATCH --cpus-per-task 16                  # request cores
#SBATCH -t 0-03:00                          # runtime in D-HH:MM format
#SBATCH -p short	                        # partition to run in
#SBATCH --mem=32G                           # memory for all cores
#SBATCH -o /n/data1/hms/dbmi/zitnik/lab/users/jop1090/graphsearch/data/slurm/run_benchmark_%j.out   # file to which STDOUT will be written, including job ID (%j)
#SBATCH -e /n/data1/hms/dbmi/zitnik/lab/users/jop1090/graphsearch/data/slurm/run_benchmark_%j.err   # file to which STDERR will be written, including job ID (%j)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jtpolonuer@gmail.com

export PROJECT_DIR="/n/data1/hms/dbmi/zitnik/lab/users/jop1090/graphsearch"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

python src/experiments/subgraph_explorer.py --graph_name prime
#!/bin/bash 
#SBATCH	-t 1-00:00
#SBATCH -p gpu
#SBATCH --mem 20000
#SBATCH -c 1 
#SBATCH --gres=gpu 
#SBATCH  -o ../../slurm_output/cnn.out
#SBATCH -e ../../slurm_output/cnn.err
#SBATCH --mail-user=noahmcneal@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

RAW_DATA_DIR="/n/hetgfs1/VAEMetrics/kfraser"
OUT_DATA_ROOT="/n/netscratch/reece_lab/Lab/noah/data_from_training_cnn"

module load python/3.10.13-fasrc01
source activate pt2.3.0_cuda12.1

python -c 'from generate_and_train_kappa import run_all_kappa as func; func("/n/hetgfs1/VAEMetrics/kfraser", 10, 1000, "/n/netscratch/reece_lab/Lab/noah/data_from_training_cnn")'

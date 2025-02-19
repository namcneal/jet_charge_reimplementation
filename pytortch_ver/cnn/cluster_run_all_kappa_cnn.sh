#!/bin/bash 
#SBATCH	-t 1-00:00
#SBATCH -p gpu_requeue
#SBATCH --mem 20000
#SBATCH -c 1 
#SBATCH --gres=gpu 
#SBATCH  -o ../../slurm_output/cnn.out
#SBATCH -e ../../slurm_output/cnn.err
#SBATCH --mail-user=noahmcneal@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

RAW_DATA_DIR=/n/home02/nmcneal/jet_tagging/data
OUT_DATA_ROOT=/n/netscratch/reece_lab/Lab/noah/data_from_training_cnn

module load python/3.10.13-fasrc01
source activate pt2.3.0_cuda12.1

COMMAND=$(python -c "from generate_and_train_kappa import run_all_kappa as f; print(f('$RAW_DATA_DIR', 10, 1000, '$OUT_DATA_ROOT'))")
# python -c 'from generate_and_train_kappa import run_all_kappa as func; func($RAW_DATA_DIR, 10, 1000, $OUT_DATA_DIR)'

# scp  nmcneal@login.rc.fas.harvard.edu:'/n/netscratch/reece_lab/Lab/noah/data_from_training_cnn/saved-data-for-1000GeV-*-tenth-kappa/roc_curve_for*.png'  S:\\home\\documents\\research\\jet_tagging\\results_from_cluster
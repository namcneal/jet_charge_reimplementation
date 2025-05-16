#!/bin/bash 
#SBATCH	-t 3-00:00
#SBATCH -p gpu_requeue
#SBATCH --mem 20000
#SBATCH -c 1 
#SBATCH --gres=gpu 
#SBATCH  -o ../../slurm_output/cnn.out
#SBATCH -e ../../slurm_output/cnn.err
#SBATCH --mail-user=noahmcneal@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

MIN_SEED=1
MAX_SEED=10
DATA_YEAR=2017
DATA_ENERGY_GEV=1000
NUM_EPOCHS=1

RAW_DATA_DIR=/n/home02/nmcneal/jet_tagging/data
OUT_JET_IMAGE_DIR=/n/netscratch/reece_lab/Lab/noah/data_from_training_cnn
SAVE_DATA_DIR=/n/home02/nmcneal/jet_tagging/results

module load python/3.10.13-fasrc01
source activate pt2.3.0_cuda12.1

PYTHON_SCRIPT=main.py
python $PYTHON_SCRIPT --raw-jet-data-dir $RAW_DATA_DIR --min-data-seed $MIN_SEED --max-data-seed $MAX_SEED --data-year $DATA_YEAR --energy-gev $DATA_ENERGY_GEV --image-dir $OUT_JET_IMAGE_DIR --save-dir $SAVE_DATA_DIR --num-epochs $NUM_EPOCHS

# For downloading from the cluster:
# scp  nmcneal@login.rc.fas.harvard.edu:'/n/netscratch/reece_lab/Lab/noah/data_from_training_cnn/saved-data-for-1000GeV-*-tenth-kappa/roc_curve_for*.png'  S:\\home\\documents\\research\\jet_tagging\\results_from_cluster
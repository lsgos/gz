#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --job-name="gz"
#SBATCH --array=0-320%8
#SBATCH	--exclude="oat1"
export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export TMPDIR=/scratch/${USER}/tmp
export PYTHONPATH=$PYTHONPATH:~/gz_round2/gz/batchbald_redux
mkdir -p $TMPDIR
BUILD_DIR=/scratch-ssd/${USER}/conda_envs/pip-build
/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f gz_by_hand.yaml
source /scratch-ssd/oatml/miniconda3/bin/activate gz_by_hand
# manually install a version of this package without the stupid progress bars everywhere.

#pip -e batchbald_redux
ACQUISITIONS=(random BALD)
ENCODER=(True False)
DATA_AUG=(True False)
SPATIAL_TRANSFORMER=(True False)
SPATIAL_VAE=(True False)
BAR_NO_BAR=(False True)
N_SEEDS=5
N_ACQUISITIONS=${#ACQUISITIONS[@]}
N_ENCODER=${#ENCODER[@]}
N_BAR=${#BAR_NO_BAR[@]}
N_DATA_AUG=${#DATA_AUG[@]}
N_SPATIAL_TRANSFORMER=${#SPATIAL_TRANSFORMER[@]}
N_SPATIAL_VAE=${#SPATIAL_VAE[@]}
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
ACQUISITIONS_IDX=$(( (SLURM_ARRAY_TASK_ID / N_SEEDS) % N_ACQUISITIONS ))
ENCODER_IDX=$(( (SLURM_ARRAY_TASK_ID / (N_ACQUISITIONS * N_SEEDS)) % N_ENCODER ))
BAR_IDX=$(( (SLURM_ARRAY_TASK_ID / (N_ACQUISITIONS * N_SEEDS * N_ENCODER)) % N_BAR  ))
DATA_AUG_IDX=$(( (SLURM_ARRAY_TASK_ID / (N_ACQUISITIONS * N_SEEDS * N_ENCODER * N_BAR)) % N_DATA_AUG  ))
SPATIAL_TRANSFORMER_IDX=$(( (SLURM_ARRAY_TASK_ID / (N_ACQUISITIONS * N_SEEDS * N_DATA_AUG * N_BAR * N_ENCODER)) % N_SPATIAL_TRANSFORMER  ))
SPATIAL_VAE_IDX=$(( (SLURM_ARRAY_TASK_ID / (N_ACQUISITIONS * N_SEEDS * N_DATA_AUG * N_SPATIAL_TRANSFORMER *N_BAR * N_ENCODER )) % N_SPATIAL_VAE  ))

nvidia-smi -q | grep UUID

COMPLETED_TASKS=(122 114 31 105 117 110 87 126 70 113 106 123 104 47 58 111 99 102 10 116 119 109 98 101 120 108 100 107 15 45 11 19 127 118 18 24 25 121 56 115 14 22 26 112 103)

declare -A array
for n in ${COMPLETED_TASKS[@]}
do
    array[$n]=1
done


# ENCODER must be TRUE for Spatial vae and spatial transformer
if [[ ${SPATIAL_TRANSFORMER[$SPATIAL_TRANSFORMER_IDX]} == True ]] && [[ ${ENCODER[$ENCODER_IDX]} == False ]]; then
    echo SKIPPING
elif [[ ${SPATIAL_VAE[$SPATIAL_VAE_IDX]} == True ]] && [[ ${ENCODER[$ENCODER_IDX]} == False ]]; then
    echo SKIPPING
elif [[ ${SPATIAL_VAE[$SPATIAL_VAE_IDX]} == True ]] && [[ ${SPATIAL_TRANSFORMER[$SPATIAL_TRANSFORMER_IDX]} == True ]]; then
    echo SKIPPING
elif [[  ${array[$SLURM_ARRAY_TASK_ID]}  ]]; then
     echo SKIPPING
else
    srun python -u al_test.py with 'dataset=gz' \
	 bar_no_bar=${BAR_NO_BAR[$BAR_IDX]} \
	 use_pose_encoder=${ENCODER[$ENCODER_IDX]} \
	 seed=$SEED \
	 data_aug=${DATA_AUG[$DATA_AUG_IDX]} \
	 spatial_vae=${SPATIAL_VAE[$SPATIAL_VAE_IDX]} \
	 spatial_transformer=${SPATIAL_TRANSFORMER[$SPATIAL_TRANSFORMER_IDX]} \
	 acquisition=${ACQUISITIONS[$ACQUISITIONS_IDX]} \
	 'pretrain_epochs=100' \
	 'transform_spec=["Translation", "Rotation"]' \
	 classify_from_z=False \
	 pixel_likelihood=laplace \
	 -F output_rmse_fixed
fi
 

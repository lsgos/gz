#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --job-name="gz_test"
#SBATCH --array=39%16
export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export TMPDIR=/scratch/${USER}/tmp
mkdir -p $TMPDIR
BUILD_DIR=/scratch-ssd/${USER}/conda_envs/pip-build
/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f gz_mizu.yml
source /scratch-ssd/oatml/miniconda3/bin/activate gz_mizu
# manually install a version of this package without the stupid progress bars everywhere.
pip -e batchbald_redux
ACQUISITIONS=(random BALD)
ENCODER=(True)
BAR_NO_BAR=(False)
N_SEEDS=1
N_ACQUISITIONS=${#ACQUISITIONS[@]}
N_ENCODER=${#ENCODER[@]}
N_BAR=${#BAR_NO_BAR[@]}
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
ACQUISITIONS_IDX=$(( (SLURM_ARRAY_TASK_ID / N_SEEDS) % N_ACQUISITIONS ))
ENCODER_IDX=$(( (SLURM_ARRAY_TASK_ID / (N_ACQUISITIONS * N_SEEDS)) % N_ENCODER ))
BAR_IDX=$(( (SLURM_ARRAY_TASK_ID / (N_ACQUISITIONS * N_SEEDS * N_ENCODER)) % N_BAR  ))
srun python -u al_test.py with 'dataset=gz' \
                               bar_no_bar=${BAR_NO_BAR[$BAR_IDX]} \
                               use_pose_encoder=${ENCODER[$ENCODER_IDX]} \
                               seed=$SEED \
                               acquisition=${ACQUISITIONS[$ACQUISITIONS_IDX]} \
                               'pretrain_epochs=100' \
                               'transform_spec=["Translation", "Rotation"]' \
                               classify_from_z=False \
                               pixel_likelihood=laplace \
                               -F output_rmse_fixed

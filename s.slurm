#!/bin/bash
#SBATCH -A m4750_g # project to charge for this job
#SBATCH -N 6 # number of nodes
#SBATCH -C gpu # type of architecture (gpu or cpu)
#SBATCH -q regular # quality of service (regular, debug, ...)
#SBATCH -J swift-test # name of job
#SBATCH --mail-user=dowonkim1013@gmail.com # mail user if state of job changes
#SBATCH --mail-type=ALL
#SBATCH -t 5:00:00 # max walltime
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"

cd /global/homes/d/dowon/infant-fmri
#source /usr/anaconda3/etc/profile.d/conda.sh
module load conda
conda activate py39

echo | pip freeze | grep torch

# Check if PyTorch can access the GPUs
python -c "import torch; print(torch.cuda.is_available())"
 
echo "Available environments:"
conda env list
 
TRAINER_ARGS='--accelerator gpu --max_epochs 10 --precision 16 --num_nodes 6 --devices 4 --strategy DDP' # specify the number of gpus as '--devices'
MAIN_ARGS='--loggername neptune --classifier_module v6 --dataset_name dHCP --image_path /pscratch/sd/d/dowon/dHCP/fmriprep/1.rs_fmri/8.dHCP_MNI_to_TRs_z-norm_registered'
DATA_ARGS='--batch_size 8 --num_workers 8  --input_type rest'
DEFAULT_ARGS='--project_name SwiFT-dHCP-sex'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --clf_head_version v1 --downstream_task bsid_cog_composite --downstream_task_type regression' #--use_scheduler --gamma 0.5 --cycle 0.5'
RESUME_ARGS=''
 
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYTdlNWUzYS0yZWVhLTRlMjMtOTlmZi0zMTMxNDY5Y2RkOWMifQ==" # when using neptune as a logger
 
srun python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 1e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 2 2 2 2 --window_size 4 4 4 4 --img_size 64 64 64 20 --loggername neptune \
--patch_size 8 8 8 1 --augment_during_training 
#--load_model_path /pscratch/sd/d/dowon/SwiFT/pretrained_models/hcp_sex_classification.ckpt
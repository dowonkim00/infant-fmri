#!/bin/bash

cd /pscratch/sd/d/dowon/SwiFT
#source /usr/anaconda3/etc/profile.d/conda.sh
module load conda
conda activate py39

# Check if PyTorch can access the GPUs
python -c "import torch; print(torch.cuda.is_available())"

echo "Available environments:"
conda env list

TRAINER_ARGS='--accelerator gpu --max_epochs 5 --precision 16 --num_nodes 1 --devices 0,1,2,3 --strategy DDP' # specify the number of gpus as '--devices'
MAIN_ARGS='--loggername neptune --classifier_module v6 --dataset_name dHCP --image_path /pscratch/sd/d/dowon/dHCP/fmriprep/1.rs_fmri/7.dHCP_MNI_to_TRs_z-norm'
DATA_ARGS='--batch_size 8 --num_workers 8 --input_type rest'
DEFAULT_ARGS='--project_name SwiFT-dHCP-sex'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --clf_head_version v1 --downstream_task age' #--use_scheduler --gamma 0.5 --cycle 0.5'
RESUME_ARGS=''

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYTdlNWUzYS0yZWVhLTRlMjMtOTlmZi0zMTMxNDY5Y2RkOWMifQ==" # when using neptune as a logger

export CUDA_VISIBLE_DEVICES=0,1,2,3,4

python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 \
--sequence_length 20 --first_window_size 2 2 2 2 --window_size 4 4 4 4 --img_size 96 96 96 20 --loggername neptune \
--patch_size 6 6 6 1 --augment_during_training 
#--load_model_path /pscratch/sd/d/dowon/SwiFT/pretrained_models/contrastive_pretrained.ckpt
#--attn_drop_rate 0.2
#--load_model_path /pscratch/sd/d/dowon/SwiFT/pretrained_models/hcp_sex_classification.ckpt

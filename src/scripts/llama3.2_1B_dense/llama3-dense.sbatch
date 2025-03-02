#!/bin/bash 
#SBATCH --job-name=llama-dense
#SBATCH --time=1-00:00
#SBATCH --partition=ou_sloan_teaching
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=llama3.2-1B-dense-%j.out
#SBATCH --error=llama3.2-1B-dense-%j.err
#SBATCH --array=0-0

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
cd ../..

seeds=(42 2024 256)
seed=${seeds[$TASK_ID]}

python run_llama.py \
    --model_name_or_path /nfs/pool002/users/mmakni/Llama-3.2-1B \
    --dataset_name "c4" \
    --block_size 1024 \
    --output_dir /nfs/pool002/users/mmakni/Llama-3.2-1B_dense_sparsegpt_1127 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "steps" \
    --do_eval \
    --bf16 True \
    --lora_num_ranks 64 \
    --lora_dropout 0.0 \
    --lora_config "dense" \
    --preprocessing_num_workers 8 \
    --seqlen 2048 \
    --seed $seed \

#!/bin/bash 
#SBATCH --job-name=llama-alps
#SBATCH --time=1-00:00
#SBATCH --partition=ou_sloan_teaching
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=llama3.2_1B-scale-alps-N-M-fixed-compression-%j.out
#SBATCH --error=llama3.2_1B-scale-alps-N-M-fixed-compression-%j.err
#SBATCH --array=0-1

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
cd ../..

compression_ratios=(0.5 0.5)
compression_ratio=${compression_ratios[$TASK_ID]}

prunens=(2 3)
prunen=${prunens[$TASK_ID]}

prunems=(8 8)
prunem=${prunems[$TASK_ID]}

python run_llama.py \
    --model_name_or_path /nfs/pool002/users/mmakni/Llama-3.2-1B \
    --dataset_name "c4" \
    --block_size 1024 \
    --output_dir /nfs/pool002/users/mmakni/Llama-3.2-1B_scale_alps_${prunen}_${prunem}_${compression_ratio}_1124 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "steps" \
    --do_eval \
    --bf16 True \
    --lora_num_ranks 64 \
    --lora_dropout 0.0 \
    --lora_config "scale-alps-gd" \
    --preprocessing_num_workers 8 \
    --compression_ratio $compression_ratio \
    --unstructured_sparsity -1 \
    --rank_ratio -1 \
    --prunen $prunen \
    --prunem $prunem \
    --seqlen 2048 \
    --nsamples 128 \
    --am_iters 80 \
    --percdamp 0.005 \
    --hess_diag True\
    --hess_percdamp 0.005\
    --gd_lr_init 1e-2 \
    --gd_iters 50 \
    --seed 42 \

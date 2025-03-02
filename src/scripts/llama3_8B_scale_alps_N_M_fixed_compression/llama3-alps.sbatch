#!/bin/bash 
#SBATCH --job-name=llama-alps
#SBATCH --time=1-00:00
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=llama3_8B-scale-alps-N-M-fixed-compression-%j.out
#SBATCH --error=llama3_8B-scale-alps-N-M-fixed-compression-%j.err
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
    --model_name_or_path /nfs/pool002/users/mmakni/Meta-Llama-3-8B \
    --dataset_name "c4" \
    --block_size 1024 \
    --output_dir /nfs/pool002/users/mmakni/Meta-Llama-3-8B_scale_alps_${prunen}_${prunem}_${compression_ratio}_0225 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "steps" \
    --do_eval \
    --bf16 True \
    --tf32 True \
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
    --percdamp 0.01 \
    --hess_diag True\
    --hess_percdamp 0.1\
    --gd_lr_init 5e-3 \
    --gd_iters 100 \
    --seed 42 \

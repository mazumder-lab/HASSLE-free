#!/bin/bash 
#SBATCH --job-name=llama-sparsegpt
#SBATCH --time=1-00:00
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=llama3_8B-scale-sparsegpt-sparsity-fixed-rank-ratio-%j.out
#SBATCH --error=llama3_8B-scale-sparsegpt-sparsity-fixed-rank-ratio-%j.err
#SBATCH --array=0-2

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
cd ../..

sparsities=(0.6 0.7 0.8)
sparsity=${sparsities[$TASK_ID]}


python run_llama.py \
    --model_name_or_path /nfs/pool002/users/mmakni/Meta-Llama-3-8B \
    --dataset_name "c4" \
    --block_size 1024 \
    --output_dir /nfs/pool002/users/mmakni/Meta-Llama-3-8B_scale_sparsegpt_${prunen}_${prunem}_${compression_ratio}_1124 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "steps" \
    --do_eval \
    --bf16 True \
    --tf32 True \
    --lora_num_ranks 64 \
    --lora_dropout 0.0 \
    --lora_config "scale-sparsegpt-gd" \
    --preprocessing_num_workers 8 \
    --compression_ratio $sparsity \
    --unstructured_sparsity -1 \
    --rank_ratio 0.3 \
    --prunen 0 \
    --prunem 0 \
    --seqlen 2048 \
    --nsamples 128 \
    --am_iters 80 \
    --percdamp 0.005 \
    --hess_diag True\
    --hess_percdamp 0.005\
    --gd_lr_init 1e-2 \
    --gd_iters 50 \
    --seed 42 \

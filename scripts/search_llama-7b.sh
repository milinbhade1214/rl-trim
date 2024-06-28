#!/bin/bash
#SBATCH --job-name=Llama_AMC_train# Job name
#SBATCH --ntasks=8 # Run on a single CPU
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH --output=results/L2_importance%j.out # Standard output and error log

cuda_device=7


# # Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

python amc_search.py \
    --job=train \
    --model=Llama-2-7b-hf \
    --dataset=wikitext \
    --preserve_ratio=0.8 \
    --lbound=0.7 \
    --rbound=1.0 \
    --reward=perp_reward \
    --data_root=./dataset \
    --ckpt_path=./checkpoint \
    --model_path=/data/home/milinbhade/LLaMa/hf_llama \
    --tokenizer_path=/data/home/milinbhade/LLaMa/hf_llama \
    --seed=2024 \
    --warmup 50 \
    --discount 0.99
#!/bin/bash
#SBATCH --job-name=Llama_AMC_train# Job name
#SBATCH --ntasks=8 # Run on a single CPU
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH --output=AMC_export_LLama%j.out # Standard output and error log

cuda_device=7
RATIO=0.5
# # Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

python amc_search.py \
    --job=export \
    --model=Llama-2-7b-hf \
    --dataset=wikitext \
    --data_root=./dataset \
    --ckpt_path=./checkpoint \
    --seed=2018 \
    --n_worker=32 \
    --preserve_ratio=$RATIO \
    --pr_ratio=$RATIO \
    --lbound=0.4 \
    --rbound=1.0 \
    --ratios=1.0,0.9979106104651163,1.0,0.9991824127906976,1.0,0.9996366279069767,1.0,0.9982739825581395,1.0,0.9959120639534884,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488,0.40625,0.3999818313953488 \
    --model_path=/data/home/milinbhade/LLaMa/hf_llama \
    --tokenizer_path=/data/home/milinbhade/LLaMa/hf_llama \
    --export_path=./exported_models/ \
    --export_num=2648 \
    


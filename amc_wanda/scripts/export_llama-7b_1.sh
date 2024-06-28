#!/bin/bash
#SBATCH --job-name=Llama_AMC_train# Job name
#SBATCH --ntasks=8 # Run on a single CPU
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH --output=AMC_export_LLama%j.out # Standard output and error log

cuda_device=1
RATIO=0.6
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
    --lbound=0.5 \
    --rbound=1.0 \
    --ratios=0.96875,0.9916424418604651,0.96875,0.9513989825581395,1.0,0.9925508720930233,0.96875,0.9870094476744186,1.0,0.973655523255814,1.0,0.965843023255814,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5 \
    --model_path=/data/home/milinbhade/LLaMa/hf_llama \
    --tokenizer_path=/data/home/milinbhade/LLaMa/hf_llama \
    --export_path=./exported_models/ \
    --export_num=2642 \
    


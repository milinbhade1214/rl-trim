#!/bin/bash
#SBATCH --job-name=Train_BERT # Job name
#SBATCH --ntasks=8 # Run on a single CPU
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH --output=rte_finetune%j.out # Standard output and error log

 
python finetune.py
echo "FinishedTraining"
python -W ignore amc_fine_tune.py \
    --model=bert-base-uncased \
    --dataset=cola \
    --lr=0.05 \
    --n_gpu=4 \
    --batch_size=256 \
    --n_worker=32 \
    --lr_type=cos \
    --n_epoch=150 \
    --wd=4e-5 \
    --seed=2018 \
    --data_root=./dataset/ \
    --tokenizer_path=./checkpoint \
    --model_path=./checkpoint \
    --finetuned_dir=./finetuned_models \
    --ckpt_path=./exported_models/bert-base-uncased_mrpc_1.0_exported.pth \
    --prune_heads='{"0": [0], "1": [7], "2": [11], "3": [1, 2, 3, 4, 6, 7, 8, 11], "4": [0, 1, 2, 4, 5, 6, 8, 9, 10, 11], "5": [8], "6": [1, 2, 5, 6, 7, 8, 9], "7": [], "8": [], "9": [2, 5], "10": [0, 1, 2, 3, 4, 6, 7, 11], "11": [0, 2, 6, 7, 9, 10, 11]}'
    
       

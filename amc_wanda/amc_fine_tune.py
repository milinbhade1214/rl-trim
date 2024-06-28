# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu




import time
import argparse
import shutil
import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter

from lib.utils import accuracy, AverageMeter, progress_bar, get_output_folder
from lib.data import get_dataset
from lib.net_measure import measure_model
from transformers import BertForSequenceClassification
import json


checkpoint = {
    'cola' : 'Ruizhou/bert-base-uncased-finetuned-cola',
    'mnli': 'textattack/bert-base-uncased-MNLI',        ##### ------> Not a good checkpoint
    'mrpc': 'Intel/bert-base-uncased-mrpc',
    'qqp': 'textattack/bert-base-uncased-QQP',
    'rte': 'anirudh21/bert-base-uncased-finetuned-rte',
    'sst2': 'doyoungkim/bert-base-uncased-finetuned-sst2',
    'stsb': 'textattack/bert-base-uncased-STS-B',
    'wnli' : 'anirudh21/bert-base-uncased-finetuned-wnli',
    'qnli' : 'anirudh21/bert-base-uncased-finetuned-qnli',
}
checkpoint_squad = {
    'squad': 'csarron/bert-base-uncased-squad-v1',
    'squad_v2': 'IProject-10/bert-base-uncased-finetuned-squad2',
}



def parse_args():
    parser = argparse.ArgumentParser(description='AMC fine-tune script')
    parser.add_argument('--model', default='bert-base-uncased', type=str, help='name of the model to train')
    parser.add_argument('--dataset', default='mrpc', type=str, help='name of the dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--n_worker', default=4, type=int, help='number of data loader worker')
    parser.add_argument('--lr_type', default='exp', type=str, help='lr scheduler (exp/cos/step3/fixed)')
    parser.add_argument('--n_epoch', default=150, type=int, help='number of epochs to train')
    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    
    
    
    parser.add_argument('--model_path', default=None, type=str, help='model path to resume from')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    # resume
    parser.add_argument('--ckpt_path', default=None, type=str, help='checkpoint path to resume from')
    # run eval
    parser.add_argument('--eval', action='store_true', help='Simply run eval')
    parser.add_argument("--prune_heads", default=None, type=str, help="Dictionary of heads to prune")


    parser.add_argument("--tokenizer_path", default=None, type=str, help="Path to tokenizer")
    parser.add_argument("--finetuned_dir", default=None, type=str, help="Path to save finetuned models")

    return parser.parse_args()


def get_model(heads_to_prune=None, task='mrpc', args=None):
    print('=> Building model..')
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

    output_attentions = False
    output_hidden_states = False

    if args.model_path is not None:
        print("Loading model from folder")
        model_checkpoint = args.model_path + "/" + task
    else:
        model_checkpoint = checkpoint[task]
    
    net = BertForSequenceClassification.from_pretrained(model_checkpoint, 
                                                        num_labels=num_labels, 
                                                        output_attentions=output_attentions, 
                                                        output_hidden_states=output_hidden_states)
    print("Model Size before Pruning: ", net.get_memory_footprint())
    if heads_to_prune is not None:
        prune_heads = json.loads(args.prune_heads)
    print("Heads to Prune: ", prune_heads)

    heads_to_prune = {}
    for key, value in prune_heads.items():
        heads_to_prune[int(key)] = value

    for key, value in heads_to_prune.items():
        net.bert.encoder.layer[key].attention.prune_heads(value)  
    print(net)
    print("Model Size after Pruning: ", net.get_memory_footprint())

    return net.cuda() if use_cuda else net



def save_checkpoint(state, is_best, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, 'ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))

from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    for i in range(0,8):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()



if __name__ == '__main__':
    args = parse_args()

    print("Arguments: ", args)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    


    print('=> Preparing data..')
    import numpy as np


    task = args.dataset
    print("Task: ", task)

    model_dir = args.model_path + "/" + str(task)
    dataset_dir = args.data_root + "/" + str(task)
    tokenizer_dir = args.tokenizer_path + "/" + str(task)

    print("Model Path: ", model_dir)
    print("Dataset Path: ", dataset_dir)
    print("Tokenizer Path: ", tokenizer_dir)


    batch_size = 2
    from datasets import load_dataset, load_metric, load_from_disk
    actual_task = "mnli" if task == "mnli-mm" else task

    print_gpu_utilization()
    from transformers import AutoTokenizer
    import torch
    metric = load_metric('glue', actual_task)
    
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    

    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding

    ##### Dataset loading from folder or downloading
    print("Raw Dataset Loading")
    if dataset_dir is not None:
        print("Loading dataset from folder")
        raw_datasets = load_dataset("glue", actual_task)
    else:
        raw_datasets = load_from_disk(actual_task)



    
    checkpoint = "bert-base-uncased"

    ##### Tokenizer Loading from folder or downloading 
    if tokenizer_dir is not None:
        print("Loading Tokenizer from folder")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print("Tokenizer Loaded")


    sentence1_key, sentence2_key = task_to_keys[task]
    def tokenize_function(examples):    
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], padding='max_length',truncation=True, max_length=512, 
                            return_tensors="pt")
        return tokenizer(examples[sentence1_key], examples[sentence2_key], padding='max_length',truncation=True, max_length=512, 
                             return_tensors="pt")


    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'label', "token_type_ids"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=512)
    print(tokenized_datasets)

    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation']



    # train_loader, val_loader, n_class = get_dataset(args.dataset, args.batch_size, args.n_worker,
    #                                                 data_root=args.data_root)
    print("Getting models")

    print("Received Pruned List")
    print(args.prune_heads)
    # model = get_model(args.prune_heads, task, args)


    model = torch.load(args.ckpt_path + 'bert-base-uncased_mrpc_1.0_completeModel.pth') 

    print("Model Loaded")
    print(model)
    
    # print("Loading checkpoint for model")
    # model.load_state_dict(torch.load(args.ckpt_path))  ####----------------> Make it more general ckpt_path


    print("Loading Training Arguments and Trainer")
    from transformers import TrainingArguments

    training_args = TrainingArguments("test-trainer", per_device_train_batch_size=1, 
                                      gradient_accumulation_steps=8, 
                                    #   gradient_checkpointing=True, 
                                      per_device_eval_batch_size=16, 
                                      evaluation_strategy="epoch",
                                        num_train_epochs=3,              # total number of training epochs
                                        metric_for_best_model='accuracy',
                                        )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)
    from transformers import Trainer

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )



    print("Model Loaded in trainer")
    print(trainer.model)
    print("Evaluating before fine-tuning")
    result = trainer.evaluate()
    print(result)

    
    # print("Finetuning model")
    # output = trainer.train()
    # print(output)

    print("Evaluating after fine-tuning")
    result = trainer.evaluate()
    print(result)
    
    #### Save Finetuned Models in finetuned_dir folder
    if args.finetuned_dir is not None:
        finetuned_model_dir = args.finetuned_dir + "/" + str(task)
        print("Saving Finetuned Model")
        trainer.save_model(finetuned_model_dir)

        state = model.state_dict()
        is_best = True
        checkpoint_dir = "."
        model.save_checkpoint(state, is_best, checkpoint_dir='.')
        print("Finetuned Model Saved")



    print("Model Summary after pruning -----------")
    for layer in range(0,12):
        print("Layer - ", layer)
        print(trainer.model.bert.encoder.layer[layer].attention.pruned_heads)

   
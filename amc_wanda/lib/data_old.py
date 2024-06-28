# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np

import os



from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, DataCollatorWithPadding

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

# def preprocess_function(examples):
#     sentence1_key, sentence2_key = task_to_keys[task]
#     if sentence2_key is None:
#         return tokenizer(examples[sentence1_key], truncation=True)
#     return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True,  padding='max_length', max_length=512)


def get_dataset(dset_name, batch_size, n_worker, data_root='../../data'):
    

    dataset = load_dataset("glue", dset_name)

    return dataset['train'], dataset['val'], 2



def get_split_dataset(dset_name, batch_size, n_worker, val_size, data_root='../data',
                      use_real_val=False, shuffle=True):
   
    data = load_dataset('glue', 'mrpc')
    

    ##### Needs to be changes to split training dataset into train and val
    # Split the dataset into train and validation sets
    trainset = data['train']
    valset = data['train']

    n_train = len(trainset)
    n_val = len(valset)

    assert val_size < n_train
    assert val_size < n_val

    indices = list(range(n_train))
    np.random.shuffle(indices)
    assert val_size < n_train
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    # Initialize the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Define a function to tokenize a batch of inputs
    def tokenize(batch):
        return tokenizer(batch['sentence1'], batch['sentence2'], padding='max_length', truncation=True)

    # Tokenize the datasets
    trainset = trainset.map(tokenize, batched=True, batch_size=len(trainset))
    valset = valset.map(tokenize, batched=True, batch_size=len(valset))

    # Set the format of the datasets to PyTorch
    trainset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    valset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create data collator to handle padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                               collate_fn=data_collator, num_workers=n_worker, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=val_sampler,
                                             collate_fn=data_collator, num_workers=n_worker, pin_memory=True)

    n_class = 2  # MRPC is a binary classification task

    return train_loader, val_loader, n_class
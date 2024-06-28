# Import necessary modules
import time
import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

from collections import defaultdict
import fnmatch

from tqdm import tqdm

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )
    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)
    
    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in tqdm(range(0,nsamples,bs), desc ="WikiText Validation: "):
        # if i % 50 == 0:
        #     print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


import numpy as np
from torch import autograd
# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = 15

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in tqdm(range(0,nsamples,bs), desc ="WikiText Validation: "):
#         if i % 5 == 0:
#             print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # print(device)
        device = model.model.layers[0].self_attn.o_proj.weight.device

        # print(device)

        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)

        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        with autograd.detect_anomaly():
            lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # if torch.isnan(neg_log_likelihood) or torch.isinf(neg_log_likelihood):
        #     neg_log_likelihood = torch.tensor([1e9]).to(device)  # or some large number

        # print("NLL: ", neg_log_likelihood)
        # print("Loss: ", loss)
        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # print("NLLs: ", nlls)

    # Compute perplexity
    nll_tensor = torch.stack(nlls)
    nan_mask = torch.isnan(nll_tensor)
    # print("Nan mask: ", nan_mask)
    # valid_nlls = nll_tensor[~nan_mask]
    valid_nlls = torch.where(nan_mask, torch.tensor(float(30000)), nll_tensor)  # Replace NaN values with a high value
    # print("Valid NLLs: ", valid_nlls.shape)
    # print("Valid NLLs: ", valid_nlls)
    ppl = torch.exp(valid_nlls.sum() / (nsamples * model.seqlen))

    # Compute perplexity
    # ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    

    # if np.isnan(ppl.detach().cpu().numpy()):
    #     ppl = torch.tensor([1e9])

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()
    # print("PPL: ", ppl.item())

    return ppl.item()

model_path = 'hf_llama'
def eval_zero_shot(model_name, model, tokenizer, saved_dir, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    
    print("Model passed to evaluation: ", model)
    ###### This will not work as tasks.ALL_TASKS does not exist now
    # Replace tasks.ALL_TASKS with tasks.get_task_dict().keys()
    # task_names = pattern_match(task_list, tasks.ALL_TASKS)
    task_names = pattern_match(task_list, tasks.get_task_dict(["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"]).keys())
    model_args = f"pretrained={saved_dir}"
    limit = 100
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_path}, cache_dir={model_path},use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="huggingface",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        # pretrained_model=model,
        batch_size=None,
        device=None,
        limit=limit,
    )

    return results 
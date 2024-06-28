# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import time
import torch
import torch.nn as nn
from lib.utils import AverageMeter, accuracy, prGreen
from lib.data import get_loaders 
from lib.eval import eval_ppl, eval_zero_shot, eval_ppl_wikitext

import os
from env.rewards import *
import math

import numpy as np
import copy
import json

from tqdm import tqdm

####### LLama related imports 
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaDecoderLayer, LlamaMLP, LlamaConfig


################### Wanda related imports
import time 
import heapq  

from .layerwrapper import WrappedGPT
from .data import get_loaders 




DROPOUT_FLOPS = 4

# compute mean activation (sum), computate variance of activation
# (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5




################################################### Code for Wanda Pruning ########################################################


############ Calibration input size = 10
def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']   
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

################################################### Code for Wanda Pruning ########################################################


from pynvml import *
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def find_pruneable_heads_and_indices(
    heads, n_heads, head_size, already_pruned_heads = set()
):
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index

def prune_linear_by_mask(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Masking based Pruning of linear layer, just make entries zeros
    Not change size of the matrices

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    mask = torch.zeros_like(layer.weight)
    if dim == 0:
        mask[index] = 1
    else:
        mask[:, index] = 1

    layer.weight.data *= mask
    layer.weight.requires_grad = True

    if layer.bias is not None:
        bias_mask = torch.zeros_like(layer.bias)
        if dim == 0:
            bias_mask[index] = 1
        else:
            bias_mask = 1
        layer.bias.data *= bias_mask
        layer.bias.requires_grad = True

    return layer

def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    assert max(index) < layer.weight.size(dim), "Index out of bounds"
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

### Using for resetting model
def get_llm(model_name, cache_dir="./hf_llama"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

class ChannelPruningEnv:
    def __init__(self, model, checkpoint, data, preserve_ratio, args, n_data_worker=4,
                 batch_size=256, export_model=False, use_new_input=False, text_writer = None):
        #default setting
        self.prunable_layer_types = [LlamaSdpaAttention, LlamaMLP]              ############## See if needs to be changes here

        #save options
        self.model = model
        self.checkpoint = checkpoint
        self.model_name = 'Llama-2-7b-hf'
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data
        self.preserve_ratio = preserve_ratio
        self.pr_ratio = preserve_ratio

        self.text_writer = text_writer # for recording text info

        #options from args
        self.args = args
        self.lbound = args.lbound
        self.rbound = args.rbound
        

        ###########
        ####### Just to get last pruning dict after export 
        self.pruning_dict = {}

        ##########
        self.use_real_val = args.use_real_val
        self.n_calibration_batches = args.n_calibration_batches
        self.n_points_per_layer = args.n_points_per_layer
        self.channel_round = args.channel_round
        self.acc_metric = args.acc_metric
        self.data_root = args.data_root

        self.export_model = export_model
        self.use_new_input = use_new_input

        self.export_num = args.export_num

        # sanity check
        assert self.preserve_ratio > self.lbound, 'Error! You can make achieve preserve_ratio smaller than lbound!'



        self.text_writer.write('=> Loading dataset: {}'.format(data))

        """
            Need to load wikitext or ptb dataset from train for getting proxy ppl
         
        """

        self.model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=False)
        
        self.train_loader, self.val_loader, self.test_loader = get_loaders("wikitext2", nsamples=128, seed=1234, seqlen=self.model.seqlen, tokenizer=tokenizer)    ###### This also gives validation set and a subset from the training set of size 128
        self.text_writer.write('=> Load dataset from: {}'.format("wikitext2"))

        self.calibration_data = self.train_loader    #### For wanda


        self.val_data = self.val_loader   ###-> this will be used for fast validation           -------------> Use this after actor step
        self.test_data = self.test_loader  ###-> this will be used for final testing            -------------> Use this after final pruning

        ###### Imp : Self.train_loader will be used for calibration purpose in wanda pruning

        print("Building Index")
        self.text_writer.write('=> Building index')
        self._build_index()   ## ------------------------------------->  Done 
        
        
        self.n_prunable_layer = len(self.prunable_idx)
        print('=> Prunable layers:', self.n_prunable_layer)
        self.text_writer.write('=> Prunable layers: {}'.format(self.n_prunable_layer))
        # print(self.n_prunable_layer)

        # extract information for preparing
        print("Extracting Layer Information")
        self._extract_layer_information()
        self.text_writer.write('=> Extracting layer information')

        
        # build embedding (static part)
        print("Building State Embedding")
        self._build_state_embedding()
        self.text_writer.write('=> Building state embedding')


        ##################################################################################################
        self.ignored_layers = [0,1,2,3,31]
        ######################################## Preparation for Wanda Pruning
        self.nsamples = 128

        print("Preparing Calibration Input from wikitext for wanda")
        self.use_cache = self.model.config.use_cache 
        self.model.config.use_cache = False 
        
        device=torch.device("cuda:0")
        self.device = device
        print("\n\n************************  Preparing Calibration Input from wikitext for wanda  ************************\n\n")
        self.dataloader, _, _= get_loaders("wikitext2",nsamples=10,seed=args.seed, seqlen=self.model.seqlen, tokenizer=tokenizer)
        print("dataset loading complete")


        with torch.no_grad():           ############ This needs to be done in reset also
            self.inps, self.outs, self.attention_mask, self.position_ids = prepare_calibration_input(self.model, self.dataloader, device)




        ##########################################

        # build reward
        self.reset()  # restore weight
        self.org_acc = self._validate(self.val_loader, self.model)
        print('=> original acc: {:.3f}%'.format(self.org_acc))
        self.text_writer.write('=> Original accuracy: {:.3f}%'.format(self.org_acc))


        self.org_model_size = sum(self.wsize_list)
        print('=> original weight size: {:.4f} M param'.format(self.org_model_size * 1. / 1e6))
        self.text_writer.write('=> Original weight size: {:.4f} M param'.format(self.org_model_size * 1. / 1e6))

        self.org_flops = sum(self.flops_list)
        print('=> FLOPs:')
        self.text_writer.write('=> FLOPs:')
        print([self.layer_info_dict[idx]['flops']/1e6 for idx in sorted(self.layer_info_dict.keys())])
        self.text_writer.write(json.dumps([self.layer_info_dict[idx]['flops']/1e6 for idx in sorted(self.layer_info_dict.keys())]))
        print('=> original FLOPs: {:.4f} M'.format(self.org_flops * 1. / 1e6))
        self.text_writer.write('=> Original FLOPs: {:.4f} M'.format(self.org_flops * 1. / 1e6))


        self.expected_preserve_computation = self.preserve_ratio * self.org_flops
        print('=> expected preserved computation: {:.4f} M'.format(self.expected_preserve_computation * 1. / 1e6))
        self.text_writer.write('=> Expected preserved computation: {:.4f} M'.format(self.expected_preserve_computation * 1. / 1e6))

        self.reward = eval(args.reward)

        self.best_reward = -math.inf
        self.best_strategy = None
        self.best_d_prime_list = None

        self.org_w_size = sum(self.wsize_list)
        print('=> original weight size: {:.4f} M param'.format(self.org_w_size * 1. / 1e6))
        self.text_writer.write('=> Original weight size: {:.4f} M param'.format(self.org_w_size * 1. / 1e6))  

    def step(self, action):

        if self.visited[self.cur_ind]:
            # print("Visited: ", self.cur_ind)
            action = self.strategy_dict[self.prunable_idx[self.cur_ind]][0]
            preserve_idx = self.index_buffer[self.cur_ind]
        else:
            # print("Not Visited: ", self.cur_ind)
            action = self._action_wall(action)  # percentage to preserve
            preserve_idx = None
        
        action, d_prime, preserve_idx = self.prune_kernel(self.prunable_idx[self.cur_ind], action, preserve_idx)
        # print("Inside step function after prune kernel: ", action, d_prime, preserve_idx)
        # print("Preserve Index type: ", type(preserve_idx))

        self.visited[self.cur_ind] = True
        if preserve_idx is not None:
            self.index_buffer[self.cur_ind] = preserve_idx.clone()
        else: 
            preserve_idx = torch.Tensor([])
        if self.export_model:  # export checkpoint
            print('# Pruning {}: ratio: {}, d_prime: {}\n'.format(self.cur_ind, action, d_prime))
            self.text_writer.write('# Pruning {}: ratio: {}, d_prime: {}\n'.format(self.cur_ind, action, d_prime))

        self.strategy.append(action)  # save action to strategy
        self.d_prime_list.append(d_prime)



        ###################
        """
        In case of conv layer, strategy dict consist for input and output channel pruning
        but in case of head pruning only use the first index

        
        """
        self.strategy_dict[self.prunable_idx[self.cur_ind]][0] = action
        if self.cur_ind > 0:
            self.strategy_dict[self.prunable_idx[self.cur_ind - 1]][1] = action


        ################ If last layer
        if self._is_final_layer():
            assert len(self.strategy) == len(self.prunable_idx)
            current_flops = self._cur_flops()
            acc_t1 = time.time()


            if self.export_model:
                num = self.export_num
                file_name = 'pruning_dict_{}_{}_{}_chat_{}.json'.format(self.pr_ratio, self.lbound, self.rbound, num)
                path = os.path.join('./pruning_dicts/', file_name)
                with open(path, 'w') as f:
                    json.dump(self.pruning_dict, f, indent=4)
                print("Saving pruning dict to ", path)



            acc = self._validate(self.val_loader, self.model)
            acc_t2 = time.time()
            self.val_time = acc_t2 - acc_t1
            compress_ratio = current_flops * 1. / self.org_flops
            info_set = {'compress_ratio': compress_ratio, 'accuracy': acc, 'strategy': self.strategy.copy()}
            print("Perplexity after final layer : ", acc)
            print("Current Flops before reward: ", current_flops)
            # reward = self.reward(self, acc, current_flops)      
            reward = self.reward(self, acc, current_flops, self.pr_ratio, self.org_flops)

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy.copy()
                self.best_d_prime_list = self.d_prime_list.copy()
                prGreen('New best reward: {:.4f}, acc: {:.4f}, compress: {:.4f}'.format(self.best_reward, acc, compress_ratio))
                prGreen('New best policy: {}'.format(self.best_strategy))
                prGreen('New best d primes: {}'.format(self.best_d_prime_list))

                self.text_writer.write('New best reward: {:.4f}, acc: {:.4f}, compress: {:.4f}\n'.format(self.best_reward, acc, compress_ratio))
                self.text_writer.write('New best policy: {}\n'.format(self.best_strategy))
                self.text_writer.write('New best d primes: {}\n'.format(self.best_d_prime_list))

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            if self.export_model:  # export state dict

                


                # print(self.model_name, self.data_type, "\t Preserve ratio: ", self.preserve_ratio)
                # torch.save(self.model.state_dict(), self.export_path + "{}_{}_{}_exported.pth".format(self.model_name, self.data_type, self.pr_ratio))                 #-----------------> save model to model path 
 

                ######## Do this later
                for i, module in enumerate(self.model.modules()):
                    if isinstance(module, LlamaSdpaAttention) or isinstance(module, LlamaMLP):
                        module.forward = module.old_forward
                        del module.old_forward

                # print("Saving model checkpoint at: ", self.export_path)   ##### ----------------------> Save model to model path
                # torch.save(self.model, self.export_path + "{}_{}_{}_completeModel.pth".format(self.model_name, self.data_type, self.pr_ratio, ))                 #-----------------> save model to model path'')
                return None, None, None, None
            return obs, reward, done, info_set
        info_set = None
        reward = 0
        done = False
        self.visited[self.cur_ind] = True  # set to visited
        self.cur_ind += 1  # the index of next layer
        # build next state (in-place modify)
        self.layer_embedding[self.cur_ind][-3] = self._cur_reduced() * 1. / self.org_flops  # reduced
        self.layer_embedding[self.cur_ind][-2] = sum(self.flops_list[self.cur_ind + 1:]) * 1. / self.org_flops  # rest
        self.layer_embedding[self.cur_ind][-1] = self.strategy[-1]  # last action
        obs = self.layer_embedding[self.cur_ind, :].copy()

        return obs, reward, done, info_set


    def reset(self):
        # restore env by loading the checkpoint
        print_gpu_utilization()
        # release unused memory
        torch.cuda.empty_cache()

        # self.model = get_llm(self.checkpoint, self.checkpoint)
        print("Trying to load using pretrained")
        # self.model.from_pretrained(self.checkpoint, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
        
        from transformers.modeling_utils import load_sharded_checkpoint
        load_sharded_checkpoint(self.model, self.checkpoint, 'cuda')
        # print('=> Load checkpoint from: {}'.format(self.checkpoint))
        # print(self.model)

        self.cur_ind = 0
        self.strategy = []  # pruning strategy
        self.d_prime_list = []
        self.strategy_dict = copy.deepcopy(self.min_strategy_dict)
        # reset layer embeddings
        self.layer_embedding[:, -1] = 1.
        self.layer_embedding[:, -2] = 0.
        self.layer_embedding[:, -3] = 0.
        obs = self.layer_embedding[0].copy()
        obs[-2] = sum(self.wsize_list[1:]) * 1. / sum(self.wsize_list)
        self.extract_time = 0
        self.fit_time = 0
        self.val_time = 0
        # for share index
        self.visited = [False] * len(self.prunable_idx)
        self.index_buffer = {}

        with torch.no_grad():           ############ This needs to be done in reset also
            self.inps, self.outs, self.attention_mask, self.position_ids = prepare_calibration_input(self.model, self.dataloader, self.device)



        return obs
        

    def set_export_path(self, path):
        self.export_path = path

    def prune_kernel(self, op_idx, preserve_ratio, preserve_idx=None):
        
        layers = self.model.model.layers
        '''Return the real ratio'''
        m_list = list(self.model.modules())
        op = m_list[op_idx]


        ######### Needs layer number for wanda pruning
        if isinstance(op, LlamaSdpaAttention):
            layer_num = op.layer_idx
        else:
            layer_num = m_list[op_idx-6].layer_idx

        layer = layers[layer_num]
        subset = find_layers(layer)

        if f"model.layers.{layer_num}" in self.model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = self.model.hf_device_map[f"model.layers.{layer_num}"]
            inps, outs, attention_mask, position_ids = self.inps.to(dev), self.outs.to(dev), self.attention_mask.to(dev), self.position_ids.to(dev)

        
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
        
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(self.nsamples):
            with torch.no_grad():
                self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask, position_ids=self.position_ids)[0]
        # print("Reached here")
        for h in handles:
            h.remove()

        if layer in self.ignored_layers:
            # keep_dict_head[i] = torch.arange(0,4096)
            # keep_dict_inter[i] = torch.arange(0,11008)  ########### Add the logic to add indices to the pruning_dict

            for j in range(self.nsamples):
                with torch.no_grad():
                    self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask, position_ids=self.position_ids)[0]
            self.inps, self.outs = self.outs, self.inps
            if isinstance(op, LlamaSdpaAttention):
                self.pruning_dict[op_idx] = torch.arange(0,num_heads*128).tolist()
            elif isinstance(op, LlamaMLP):
                self.pruning_dict[op_idx] = torch.arange(0,num_heads).tolist()
            else:
                print("Some sort of error")
            return 1., num_heads, None  # TODO: should be a full index
        










        ####################################################       

        assert (preserve_ratio <= 1.)
        # print("Preserve Ratio: ", preserve_ratio)
        if isinstance(op, LlamaSdpaAttention):
            num_heads = op.num_heads
        elif isinstance(op, LlamaMLP):
            num_heads = op.gate_proj.out_features
        else:
            raise NotImplementedError
        

        if preserve_ratio == 1:  # do not prune
            if isinstance(op, LlamaSdpaAttention):
                self.pruning_dict[op_idx] = torch.arange(0,num_heads*128).tolist()
            elif isinstance(op, LlamaMLP):
                self.pruning_dict[op_idx] = torch.arange(0,num_heads).tolist()
            else:
                print("Some sort of error")
            return 1., num_heads, None  # TODO: should be a full index
            # n, c, h, w = op.weight.size()
            # mask = np.ones([c], dtype=bool)

        def format_rank(x):
            rank = int(np.around(x))
            out = max(rank, 1)
            # print("Rank: ", out)
            return out 
        
        
        heads_frac = num_heads * preserve_ratio
        # print("Head Frac: ", heads_frac)
        d_prime = format_rank(heads_frac)
        d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)
        if d_prime > num_heads:
            d_prime = int(np.floor(num_heads * 1. / self.channel_round) * self.channel_round)
        extract_t1 = time.time()
        if self.use_new_input:  # this is slow and may lead to overfitting
            self._regenerate_input_feature()
        # print("Dprime: ", d_prime)
        self.text_writer.write('Dprime: {}\n'.format(d_prime))
        if isinstance(op, LlamaSdpaAttention):
            q_weights = layer.self_attn.q_proj.weight.data
            k_weights = layer.self_attn.k_proj.weight.data
            v_weights = layer.self_attn.v_proj.weight.data
            o_weights = layer.self_attn.o_proj.weight.data


            #### Scalers for QKVO matrices
            q_scaler_row = wrapped_layers['self_attn.q_proj']
            k_scaler_row = wrapped_layers['self_attn.k_proj']
            v_scaler_row = wrapped_layers['self_attn.v_proj']
            o_scaler_row = wrapped_layers['self_attn.o_proj']

            ##### Scalers for Intermediate Dim 
            gate_scaler_row = wrapped_layers['mlp.gate_proj']
            up_scaler_row = wrapped_layers['mlp.up_proj']
            down_scaler_row = wrapped_layers['mlp.down_proj']




        elif isinstance(op, LlamaMLP):
            ## Get sum of each row 
            gate_weights = layer.mlp.gate_proj.weight.data
            up_weights = layer.mlp.up_proj.weight.data
            down_weights = layer.mlp.down_proj.weight.data
            down_weights = down_weights.t()


            ##### Scalers for Intermediate Dim 
            gate_scaler_row = wrapped_layers['mlp.gate_proj']
            up_scaler_row = wrapped_layers['mlp.up_proj']
            down_scaler_row = wrapped_layers['mlp.down_proj']

    

        extract_t2 = time.time()
        self.extract_time += extract_t2 - extract_t1


        fit_t1 = time.time()
        ###Change criteria for Importance to some more robust
        if preserve_idx is None:
            # compute the importance score for LlamaSdpaAttention layer
            if isinstance(op, LlamaSdpaAttention):
                # print("For LlamaSdpaAttention")
                self.text_writer.write("For LlamaSdpaAttention\n")
                # importance = torch.from_numpy(np.abs(query_weight) + np.abs(key_weight) + np.abs(value_weight) + np.abs(out_weight)).view(32, -1).sum(dim=1)             ######### Change this to some other criteria
                head_importance = torch.abs(q_weights) * torch.sqrt(q_scaler_row.scaler_row.reshape((1,-1))) + torch.abs(k_weights) * torch.sqrt(k_scaler_row.scaler_row.reshape((1,-1))) + torch.abs(v_weights) * torch.sqrt(v_scaler_row.scaler_row.reshape((1,-1))) #+ torch.abs(o_weights) * torch.sqrt(o_scaler_row.scaler_row.reshape((1,-1)))
                head_importance = head_importance.mean(axis=0).reshape(-1, 128).sum(dim=1).detach().cpu().numpy()


                # print("Importance: ",importance)
                self.text_writer.write('Importance: {}\n'.format(head_importance))
                sorted_idx = np.argsort(-head_importance)
                # print(sorted_idx)
                
                
                self.text_writer.write('Sorted Index: {}\n'.format(sorted_idx))
                preserve_idx = sorted_idx[:d_prime]
                prune_idx = sorted_idx[d_prime:]
                # print("Preserve Index:   ", preserve_idx)
                self.text_writer.write('Preserve Index: {}\n'.format(preserve_idx))
                # print("Prune Index: ", prune_idx)
                self.text_writer.write('Prune Index: {}\n'.format(prune_idx))

            # computer importance score for LlamaMLP layer
            elif isinstance(op, LlamaMLP):
                # print("For LlamaMLP")
                self.text_writer.write("For LlamaMLP\n")
                ## Importance score = sum of each row for LlamaMLP + BertOutput

                # importance = torch.from_numpy(np.abs(gate_weight) + np.abs(up_weight) + np.abs(down_weight)).sum(dim=1)
                #### This is not correct
                gate_imp = torch.abs(gate_weights) * torch.sqrt(gate_scaler_row.scaler_row.reshape((1,-1)))
                # print("Gate Importance: ", gate_imp.shape)
                up_imp = torch.abs(up_weights) * torch.sqrt(up_scaler_row.scaler_row.reshape((1,-1)))
                # print("Up Importance: ", up_imp.shape)
                down_imp = torch.abs(down_weights) * torch.sqrt(down_scaler_row.scaler_row.reshape((-1,1)))
                # print("Down Importance: ", down_imp.shape)


                mlp_importance =  gate_imp + up_imp + down_imp
                mlp_importance = mlp_importance.mean(axis=1).detach().cpu().numpy()


                # print("Importance: ",importance)
                self.text_writer.write('Importance: {}\n'.format(mlp_importance))
                sorted_idx = np.argsort(-mlp_importance)
                # print(sorted_idx)
                self.text_writer.write('Sorted Index: {}\n'.format(sorted_idx))
                 
                prune_idx = sorted_idx[d_prime:]
                preserve_idx = sorted_idx[:d_prime]
                # print("Preserve Index:   ", preserve_idx)
                self.text_writer.write('Preserve Index: {}\n'.format(preserve_idx))
                # print("Prune Index: ", prune_idx)
                self.text_writer.write('Prune Index: {}\n'.format(prune_idx))
            

        assert len(preserve_idx) == d_prime
        mask = np.zeros([num_heads], dtype=bool)
        mask[preserve_idx] = True

        if isinstance(op, LlamaSdpaAttention):
            heads, index = find_pruneable_heads_and_indices(prune_idx, n_heads=32, head_size=4096//32, already_pruned_heads=set())

            op.q_proj = prune_linear_by_mask(op.q_proj, index).half()
            op.k_proj = prune_linear_by_mask(op.k_proj, index).half()
            op.v_proj = prune_linear_by_mask(op.v_proj, index).half()
            op.o_proj = prune_linear_by_mask(op.o_proj, index, dim=1).half()

        if isinstance(op, LlamaMLP):
            op.gate_proj = prune_linear_by_mask(op.gate_proj, torch.LongTensor(preserve_idx)).half()
            op.up_proj = prune_linear_by_mask(op.up_proj, torch.LongTensor(preserve_idx)).half()
            op.down_proj = prune_linear_by_mask(op.down_proj, torch.LongTensor(preserve_idx), dim=1).half()


        fit_t2 = time.time()
        self.fit_time += fit_t2 - fit_t1
                                                               
        action = np.sum(mask) * 1. / len(mask)
        
        if self.export_model:
            # print('# Pruning {}: ratio: {}, d_prime: {}'.format(op_idx, action, d_prime))
            self.text_writer.write('# Pruning {}: ratio: {}, d_prime: {}\n'.format(op_idx, action, d_prime))

            zero_mask_idx = np.where(~mask)[0]
            # print("Pruning Heads at index: ", zero_mask_idx, " \t from layer ", op_idx)
            self.text_writer.write('Pruning Heads at index: {} \t from layer {}\n'.format(zero_mask_idx, op_idx))
            if isinstance(op, LlamaSdpaAttention):
                print("Performing real pruning of LlamaSdpaAttention")
                # op.q_proj = prune_linear_layer(op.q_proj, index).half()
                # op.k_proj = prune_linear_layer(op.k_proj, index).half()
                # op.v_proj = prune_linear_layer(op.v_proj, index).half()
                # op.o_proj = prune_linear_layer(op.o_proj, index, dim=1).half()
                self.pruning_dict[op_idx] = index.tolist()
            elif isinstance(op, LlamaMLP):
                # print("Perform real pruning of LlamaMLP")
                # op.gate_proj = prune_linear_layer(op.gate_proj, torch.LongTensor(preserve_idx)).half()
                # op.up_proj = prune_linear_layer(op.up_proj, torch.LongTensor(preserve_idx)).half()
                # op.down_proj = prune_linear_layer(op.down_proj, torch.LongTensor(preserve_idx), dim=1).half()

                self.pruning_dict[op_idx] = preserve_idx.tolist()

            
                
            print(self.model)
            self.text_writer.write('Model: {}\n'.format(self.model))
        for j in range(self.nsamples):
            with torch.no_grad():
                self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask, position_ids=self.position_ids)[0]
        self.inps, self.outs = self.outs, self.inps
        # print("Preserve Index: inside prune function", preserve_idx)
        # self.text_writer.write('Preserve Index: inside prune function {}'.format(preserve_idx))

        preserve_idx = torch.from_numpy(preserve_idx)
        return action, d_prime, preserve_idx
        
        
        

    def _is_final_layer(self):
        return self.cur_ind == len(self.prunable_idx) - 1



    def _action_wall(self, action):
        assert len(self.strategy) == self.cur_ind

        action = float(action)
        action = np.clip(action, 0, 1)

        other_comp = 0
        this_comp = 0
        # print(self.strategy_dict)
        for i, idx in enumerate(self.prunable_idx):
            # Assume 'flops' now represents the computational cost of each head
            head_cost = self.layer_info_dict[idx]['flops']
            if i == self.cur_ind:
                this_comp += head_cost * self.strategy_dict[idx][0]         ## remove self.strategy_dict[idx][0] and replace with action
            else:
                other_comp += head_cost * self.strategy_dict[idx][0]


        # print("This Comp: ", this_comp, end="| \t")
        # print("Other Comp: ", other_comp, end="| \t")


        self.expected_min_preserve = other_comp + this_comp * action
        
        
        # print("Expected Min Preserve: ", self.expected_min_preserve, end="| \t")
        
        
        max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp
        # print("Max Preserve Ratio: ", max_preserve_ratio)
        
        
        action = np.minimum(action, max_preserve_ratio)
        # print("Action Wall : (1)", action, end="| \t")
        
        
        action = np.maximum(action, self.strategy_dict[self.prunable_idx[self.cur_ind]][0])  # impossible (should be)       ### Action should always be greater than strategy_dict
        # print("(2)", action, end="| \t")
        # print("Strategy Dict: ", self.strategy_dict[self.prunable_idx[self.cur_ind]][0])


        action = np.minimum(action, self.strategy_dict[self.prunable_idx[self.cur_ind]][1])
        # print("Final Action returned: ", action)
        return action

    def _get_buffer_flops(self, idx):
        buffer_idx = self.buffer_dict[idx]
        buffer_flop = sum([self.layer_info_dict[_]['flops'] for _ in buffer_idx])
        return buffer_flop

    def _cur_flops(self):
        orig_flops = self.org_flops

        def format_rank(x):
            rank = int(np.around(x))
            out = max(rank, 1)
            return out
        # print("Prunable index")
        modules = list(self.model.modules())
        for i, idx in enumerate(self.prunable_idx):
            c, n = self.strategy_dict[idx]  # input, output pruning ratio
            h = 4096
            s = 2048
            heads = 32
            i = 11008

            hidden_reduced = format_rank(c * 32) * 128
            head_reduced = format_rank(c * 32)
            inter_reduced = format_rank(c * 11008)
            ### Check if the module is LlamaSdpaAttention or LlamaMLP
            if isinstance(modules[idx], LlamaSdpaAttention):
                block_reduced_flops = dict(
                    kqv = -3 * 2 * h * hidden_reduced,
                    attn_scores = 0,
                    attn_softmax = -SOFTMAX_FLOPS * s * head_reduced, 
                    attn_dropout = -DROPOUT_FLOPS * s * head_reduced,
                    attn_scales = -s * head_reduced,
                    attn_avg_weighted_values = 0,
                    attn_output = -2 * h * hidden_reduced,
                    attn_output_dropout = 0,
                    attn_output_residual = 0,
                    attn_output_layer_norm = 0,
                )
                orig_flops += sum(block_reduced_flops.values())

            elif isinstance(modules[idx], LlamaMLP):
                block_reduced_flops = dict(
                    gate_proj = -2 * h * inter_reduced,
                    up_proj = -2 * h * inter_reduced,
                    gate_up = -h * inter_reduced,
                    up_act = -ACTIVATION_FLOPS * inter_reduced,
                    down = -2 * inter_reduced * h,
                    down_dropout = 0,
                    down_residual = 0,
                    down_layer_norm = 0,
                )
                orig_flops += sum(block_reduced_flops.values())
        return orig_flops

    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_flops - self._cur_flops()
        return reduced

    ###### This is not required
    def _init_data(self):
        val_size = 600 if 'mrpc' in self.data_type else 3000
        self.train_loader, self.val_loader, n_class = get_split_dataset(self.data_type, self.batch_size,
                                                                        self.n_data_worker, val_size,
                                                                        data_root=self.data_root,
                                                                        use_real_val=self.use_real_val,
                                                                        shuffle=False)  # same sampling
        if self.use_real_val:  # use the real val set for eval, which is actually wrong
            print('*** USE REAL VALIDATION SET!')
        print("Train_dataloader         ", len(self.train_loader))
        print("Val_dataloader           ", len(self.val_loader))

    ######## This is correct   ---> Done 
    def _build_index(self):
        self.prunable_idx = []
        self.prunable_ops = []
        self.layer_type_dict = {}
        self.strategy_dict = {}
        self.org_channels = []
        # build index and the min strategy dict
        for i, m in enumerate(self.model.modules()):
            if i < 61 or i > 438:  ############### Leaving first four and last four layers (54, 395)
                continue
            if isinstance(m, LlamaSdpaAttention):
                # really prunable
                self.prunable_idx.append(i)
                self.prunable_ops.append(m)
                self.layer_type_dict[i] = type(m)
                
                ### Recording 
                self.org_channels.append(m.num_heads)
                self.strategy_dict[i] = [self.lbound, self.rbound]                  ######### Change right

            ############ Add Logic for LlamaMLP
            if isinstance(m, LlamaMLP):
                self.prunable_idx.append(i)
                self.prunable_ops.append(m)
                self.layer_type_dict[i] = type(m)

                self.org_channels.append(m.intermediate_size)
                self.strategy_dict[i] = [self.lbound, self.rbound]                  ######### Change right

        # self.strategy_dict[self.prunable_idx[0]][0] = 1  # modify the input
        # self.strategy_dict[self.prunable_idx[-1]][1] = 1  # modify the output

        self.min_strategy_dict = copy.deepcopy(self.strategy_dict)

        # print('=> Prunable layer idx: {}'.format(self.prunable_idx))
        self.text_writer.write('=> Prunable layer idx: {}\n'.format(self.prunable_idx))

        # print('=> Initial min strategy dict: {}'.format(self.min_strategy_dict))
        self.text_writer.write('=> Initial min strategy dict: {}\n'.format(self.min_strategy_dict))

        print('=> Prunable layer idx: ', self.prunable_idx)
        print('=> Prunable Ops: ', self.prunable_ops)
        print('=> Layer Type Dict: ', self.layer_type_dict)



        self.visited = [False] * len(self.prunable_idx)
        self.index_buffer = {}

    ############ This seems correct
    def _extract_layer_information(self):
        m_list = list(self.model.modules())

        self.data_saver = []
        self.layer_info_dict = dict()
        self.wsize_list = []
        self.flops_list = []

        from lib.utils import measure_layer_for_pruning


        # extend the forward fn to record layer info
        def new_forward(m):
            def lambda_forward(hidden_states, *args, **kwargs):
                with torch.no_grad():
                    # m.input_feat = hidden_states.clone()
                    measure_layer_for_pruning(m, hidden_states, *args, **kwargs)                         #--------------> This is not correct, verify this again
                    ################ y includes (output, attention_scores)
                    out = m.old_forward(hidden_states, *args, **kwargs)
                    # print("Output Length: " , len(out))
                    if isinstance(m, LlamaSdpaAttention):
                            #attn_output, attn_weights, past_key_value
                            y, attn_scores, _ = out
                            if attn_scores is not None:
                                m.attn_weights = attn_scores.clone()
                    elif isinstance(m, LlamaMLP):
                        y = out
                return out
            return lambda_forward

        
        ### Modify the forward of all modules
        for m in self.model.modules():
            if isinstance(m, LlamaSdpaAttention) or isinstance(m, LlamaMLP):
                m.old_forward = m.forward
                m.forward = new_forward(m)

        # now let data flow
        print('=> Extracting information...')
        with torch.no_grad():
            for i_b, data in enumerate(self.train_loader):  # use image from train set
                input_ids, attn_mask = data
                print("Batch : ", i_b)
                if i_b == self.n_calibration_batches:
                    break

                input_ids = input_ids.to(self.model.device)
                attn_mask = attn_mask.to(self.model.device)
                # inference and collect stats
                _ = self.model.forward(input_ids)

                # print("Input passed")
                if i_b == 0:  # first batch
                    print("First Batch")
                    for idx in self.prunable_idx:
                        self.layer_info_dict[idx] = dict()
                        self.layer_info_dict[idx]['params'] = m_list[idx].params
                        self.layer_info_dict[idx]['flops'] = m_list[idx].flops
                        
                        ## Needs to be uncommented if inportance calculated based on attention scores
                        # self.layer_info_dict[idx]['attention_scores'] = m_list[idx].attention_scores.cpu().numpy()
                        self.wsize_list.append(m_list[idx].params)
                        self.flops_list.append(m_list[idx].flops)

                #####################################
                """
                Instead of storing all the attention scores, try to store attention score statistics
                """
                #####################################
                # print("Reached Stacking funciton ")
                ##### Only storing attention scores
                # for idx in self.prunable_idx:
                #     attention_scores_np = m_list[idx].attention_scores.data.cpu().numpy()
                #     if 'attention_scores' not in self.layer_info_dict[idx]:
                #         self.layer_info_dict[idx]['attention_scores'] = attention_scores_np
                #     else:
                #         self.layer_info_dict[idx]['attention_scores'] = np.vstack(
                #             (self.layer_info_dict[idx]['attention_scores'], attention_scores_np))

    
    def _regenerate_input_feature(self):
        pass ##### Not needed


    def _build_state_embedding(self):
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model.modules())
        for i, ind in enumerate(self.prunable_idx):
            m = module_list[ind]
            this_state = []
            if type(m) == LlamaSdpaAttention:
                this_state.append(i)  # index
                this_state.append(2)  # layer type, 2 for LlamaSdpaAttention and 1 for LlamaMLP
                this_state.append(m.num_heads)  # number of attention heads
                this_state.append(m.head_dim)  # size of each attention head
                this_state.append(np.prod(m.q_proj.weight.size()))  # weight size of query
                this_state.append(np.prod(m.k_proj.weight.size()))  # weight size of key
                this_state.append(np.prod(m.v_proj.weight.size()))  # weight size of value

            ################## Add code for LlamaMLP and BertOutput
            if type(m) == LlamaMLP:
                this_state.append(i)  # index
                this_state.append(1)  # layer type, 2 for LlamaSdpaAttention and 1 for LlamaMLP
                this_state.append(0)  # number of attention heads
                this_state.append(0)  # size of each attention head
                this_state.append(m.up_proj.in_features)  # weight size of query
                this_state.append(m.up_proj.out_features)  # weight size of key
                this_state.append(0)  # weight size of value



            # this 3 features need to be changed later
            this_state.append(0.)  # reduced
            this_state.append(0.)  # rest
            this_state.append(1.)  # a_{t-1}
            layer_embedding.append(np.array(this_state))

        # normalize the state
        # print(layer_embedding)
        layer_embedding = np.array(layer_embedding, 'float')
        # print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding



    def _validate(self, val_loader, model, verbose=False): 
        """
            This returns perplexity on wikitext model on the passed val loader
            ----> val loader taken from train dataset
        """

        # print("Inside Validate")
        print_gpu_utilization()
        perplexity = eval_ppl_wikitext(model, val_loader, bs=1, device=None)
        return perplexity

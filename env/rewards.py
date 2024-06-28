# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# for pruning
def acc_reward(net, acc, flops):
    return acc * 0.01

def perp_reward(net, perplexity, flops, target_density=None, orig_flops=None):
    print("Perplexity in reward function ", perplexity)
    return -1 * np.log(perplexity) * np.log(flops) * 0.01


def acc_flops_reward(net, acc, flops):
    error = (100 - acc) * 0.01
    return -error * np.log(flops)

############### Reward Giving more importance to Perplexity ####################
def preserve_perplexity(net, perplexity, flops, target_density=None, orig_flops=None):
    print("Perplexity in reward function ", perplexity)
    w_p, w_f = 0.7, 0.3
    return -1 * (w_p * np.log(perplexity) * 0.01 + w_f * np.log(flops) * 0.01)


def flops_constrained_reward_fn(net, perplexity, flops, target_density=None, orig_flops=None):
    flops_normalized = 1 - flops / orig_flops
    print("Flops Normalized: ", flops_normalized)
    reward = -1 * np.log(perplexity) 
    print("Reward: ", reward)
    if flops_normalized > (target_density+0.002):
        reward = -3 - flops_normalized
    else:
        reward += 1
    print("Final reward: ", reward)
    return reward


def harmonic_mean_reward_fn(net, perplexity, flops, target_density=None, orig_flops=None):
    beta = 1
    flops_normalized = 1 - flops / orig_flops
    reward = (1 + beta**2) * (-np.log(perplexity) * 0.01) * flops_normalized / (beta**2 * flops_normalized + (-np.log(perplexity) * 0.01))
    return reward


def punish_for_flops(net, perplexity, flops, target_density=None, orig_flops=None):

    current_density = 1 - flops / orig_flops
    print("Current Density: ", current_density)
    print("Target Density: ", target_density)
    if current_density > target_density:
        reward = (target_density - current_density) * 10      
    else:
        reward = -np.log(perplexity) * 0.1
    return reward


def imp_perplexity(net, perplexity, flops, target_density=None, orig_flops=None):
    w_p, w_f = 1.0, 1.0
    return -1 * (w_p * (np.log(perplexity)**2)  + w_f * np.log(flops))

def cased_reward(net, perplexity, flops, target_density=None, orig_flops=None):
    current_density = flops / orig_flops
    base_perplexity = 6
    permissible_perplexity = 100
    linear_zone_upto = 1000

    if perplexity < permissible_perplexity:
        reward = -1 * np.log(perplexity) * (current_density**2) * 0.1
    elif(perplexity < linear_zone_upto):
        reward = -1 * np.log(perplexity) * current_density 
    else:
        reward = -1 * np.log(perplexity) * (2 + current_density) * 0.1
    return reward
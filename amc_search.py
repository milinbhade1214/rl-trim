# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import os
import numpy as np
import argparse
from copy import deepcopy
import torch
torch.backends.cudnn.deterministic = True


import time
from env.env import ChannelPruningEnv
from lib.agent import DDPG
from lib.utils import get_output_folder

from tensorboardX import SummaryWriter
import json

import matplotlib.pyplot as plt

########### Imports from Transformers
from datasets import load_dataset, load_metric, load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding


###### LLama loading imports 
from transformers import AutoTokenizer, AutoModelForCausalLM


def plot_data(output_path, all_rewards, all_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(all_rewards)
    plt.title('Rewards over episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')

    plt.subplot(122)
    plt.plot(all_accuracies)
    plt.title('Perplexity over episodes')
    plt.xlabel('Episode')
    plt.ylabel('Perplexity')

    plt.tight_layout()
    #create a path name using os 
    path = os.path.join(output_path, 'training_plot.png')
    plt.savefig(path)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='AMC search script')

    parser.add_argument('--job', default='train', type=str, help='support option: train/export')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    # env
    parser.add_argument('--model', default="Llama-2-7b-hf", type=str, help='model to prune--> give path ')
    parser.add_argument('--dataset', default='wikitext', type=str, help='dataset to use (wikitext2/PTB) for perplexity')
    
    
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    parser.add_argument('--tokenizer_path', default=None, type=str, help='tokenizer path')
    parser.add_argument('--model_path', default='./hf_llama', type=str, help='model path')

    parser.add_argument('--preserve_ratio', default=0.8, type=float, help='preserve ratio of the model')
    parser.add_argument('--lbound', default=0.1, type=float, help='minimum preserve ratio')
    parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')
    parser.add_argument('--reward', default='acc_reward', type=str, help='Setting the reward')
    parser.add_argument('--acc_metric', default='acc5', type=str, help='use acc1 or acc5')
    parser.add_argument('--use_real_val', dest='use_real_val', action='store_true')


    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
    # parser.add_argument('--pruning_method', default='cp', type=str,
    #                     help='method to prune (fg/cp for fine-grained and channel pruning)')
    # only for channel pruning
    parser.add_argument('--n_calibration_batches', default=5, type=int,
                        help='n_calibration_batches')
    parser.add_argument('--n_points_per_layer', default=10, type=int,
                        help='method to prune (fg/cp for fine-grained and channel pruning)')
    parser.add_argument('--channel_round', default=1, type=int, help='Round channel to multiple of channel_round')
    # ddpg
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')  # prev = 1e-3
    parser.add_argument('--lr_a', default=1e-3, type=float, help='learning rate for actor')  # prev = 1e-4
    parser.add_argument('--warmup', default=10, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=100, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=10000, type=int, help='memory size for each layer')   ############ Increase this
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    # noise (truncated normal distribution)
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.95, type=float,
                        help='delta decay during exploration')
    # training
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output', default='./logs', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode', default=200, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=1234, type=int, help='random seed to set')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=16, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=4, type=int, help='number of data batch size')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # export
    parser.add_argument('--ratios', default=None, type=str, help='ratios for pruning')
    parser.add_argument('--channels', default=None, type=str, help='channels after pruning')
    parser.add_argument('--export_path', default=None, type=str, help='path for exporting models')
    parser.add_argument('--use_new_input', dest='use_new_input', action='store_true', help='use new input feature')
    parser.add_argument('--pr_ratio', default=0.8, type=float, help='Just store preserve ratio for naming')
    parser.add_argument('--export_num', default=None, type=str, help='name of the file')

    return parser.parse_args()

def print_model_parameters(model):
    name_max = 0
    shape_max = 0
    total_params = 0
    param_dict = {}
    for name, param in model.named_parameters():
        name_max = max(name_max, len(name))
        shape_max = max(shape_max, len(str(param.size())))
        total_params += param.numel()
    shape_max -= 10
    print(f'{"Layer": <{name_max}}  |  {"Shape": <{shape_max}}  |  {"# of Elements": <2}  |  {"% of Params"}')
    text_writer.write(f'{"Layer": <{name_max}}  |  {"Shape": <{shape_max}}  |  {"# of Elements": <2}  |  {"% of Params"}\n')

    print("-" * (name_max + shape_max + 40))
    text_writer.write("-" * (name_max + shape_max + 40) + "\n")


    for name, param in model.named_parameters():
        print(f'{name: <{name_max}}  |  {str(param.size())[10:]: <{shape_max}}  |  {param.numel(): <14}  |  {str(round((param.numel()/total_params), 4) * 100)}%')
        text_writer.write(f'{name: <{name_max}}  |  {str(param.size())[10:]: <{shape_max}}  |  {param.numel(): <14}  |  {str(round((param.numel()/total_params), 4) * 100)}%\n')
        param_dict[name] = param.numel()
    print("-" * (name_max + shape_max + 40))
    text_writer.write("-" * (name_max + shape_max + 40) + "\n")
    return param_dict


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


def get_model_and_checkpoint(model, dataset, checkpoint_path, n_gpu, text_writer, args): 
    ## Load the model 
    model = get_llm(args.model_path, args.model_path)
    return model, args.model_path

def train(num_episode, agent, env, output):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    all_rewards = []
    all_perplexities = []

    T = []  # trajectory
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if episode <= args.warmup:
            print("Choosing random")
            action = agent.random_action()
            # action = sample_from_truncated_normal_distribution(lower=0., upper=1., mu=env.preserve_ratio, sigma=0.5)
        else:
            print("Choosing select action using observations")
            action = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        
        # all_rewards.append(reward)
        # all_perplexities.append(info['accuracy'])

        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # fix-length, never reach here
        # if max_episode_length and episode_steps >= max_episode_length - 1:
        #     done = True

        # [optional] save intermideate model
        # if episode % int(num_episode / 3) == 0:
        #     agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            print('#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f}'.format(episode, episode_reward,
                                                                                 info['accuracy'],
                                                                                 info['compress_ratio']))
            text_writer.write(
                '#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f}\n'.format(episode, episode_reward,
                                                                                 info['accuracy'],
                                                                                 info['compress_ratio']))
            final_reward = T[-1][0]
            # print('final_reward: {}'.format(final_reward))
            # agent observe and update policy
            for r_t, s_t, s_t1, a_t, done in T:
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    agent.update_policy()

            #agent.memory.append(
            #    observation,
            #    agent.select_action(observation, episode=episode),
            #    0., False
            #)

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            tfwriter.add_scalar('reward/last', final_reward, episode)
            tfwriter.add_scalar('reward/best', env.best_reward, episode)
            tfwriter.add_scalar('info/accuracy', info['accuracy'], episode)
            tfwriter.add_scalar('info/compress_ratio', info['compress_ratio'], episode)
            tfwriter.add_text('info/best_policy', str(env.best_strategy), episode)
            # record the preserve rate for each layer
            for i, preserve_rate in enumerate(env.strategy):
                tfwriter.add_scalar('preserve_rate/{}'.format(i), preserve_rate, episode)

            text_writer.write('best reward: {}\n'.format(env.best_reward))
            text_writer.write('best policy: {}\n'.format(env.best_strategy))
    plot_data(output, all_rewards, all_perplexities)
    text_writer.close()


def export_model(env, args):
    assert args.ratios is not None or args.channels is not None, 'Please provide a valid ratio list or pruned channels'
    assert args.export_path is not None, 'Please provide a valid export path'
    print("Exporting model to {}".format(args.export_path))
    env.set_export_path(args.export_path)

    print('=> Original model channels: {}'.format(env.org_channels))
    if args.ratios:
        ratios = args.ratios.split(',')
        ratios = [float(r) for r in ratios]
        assert  len(ratios) == len(env.org_channels)
        channels = [int(r * c) for r, c in zip(ratios, env.org_channels)]
        print(channels)
    else:
        channels = args.channels.split(',')
        channels = [int(r) for r in channels]
        ratios = [c2 / c1 for c2, c1 in zip(channels, env.org_channels)]
    print('=> Pruning with ratios: {}'.format(ratios))
    print('=> Channels after pruning: {}'.format(channels))

    for r in ratios:
        print("Ratio passed in export model: ", r)
        env.step(r)
    # print(env.pruning_dict)
    pruning_dict = env.pruning_dict
    print("Before creating dict")
    pruned_h = 0
    pruned = 0
    for k, v in pruning_dict.items():
        pruned_h = len(v)
        print(pruned_h)
        pruned += pruned_h

    total_units = sum(env.org_channels)
    # print(json.dumps(pruning_dict))
    text_writer.write("\n" + json.dumps(pruning_dict)+"\n")
    print("Total Pruned Heads: {} ----> {}".format(pruned_h, (pruned_h/total_units) * 100))
    text_writer.write("Total Pruned Heads: {} ----> {}\n".format(pruned, (pruned/total_units) * 100))
    return


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    import warnings
    warnings.filterwarnings('ignore')
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)


    str_export = 'export' if args.job == 'export' else 'search' 
    base_folder_name = '{}_{}_r{}_{}'.format(args.model, args.dataset, args.preserve_ratio, str_export)
    if args.suffix is not None:
        base_folder_name = base_folder_name + '_' + args.suffix
    args.output = get_output_folder(args.output, base_folder_name)
    print('=> Saving logs to {}'.format(args.output))
    tfwriter = SummaryWriter(logdir=args.output)
    text_writer = open(os.path.join(args.output, 'log.txt'), 'w')
    print('=> Output path: {}...'.format(args.output))




    ####### No need to give checkpoint as path will suffice, check in reset code
    model, checkpoint = get_model_and_checkpoint(args.model, args.dataset, checkpoint_path=args.ckpt_path,
                                                 n_gpu=args.n_gpu, text_writer=text_writer, args=args)

    print("args:")
    print(args)


    ######## Logging args and model and dataset information
    text_writer.write('Arguments: \n {}\n'.format(args))
    text_writer.write('Model: \n {}\n'.format(model))
    text_writer.write('Dataset: \n {}\n'.format(args.dataset))




    env = ChannelPruningEnv(model, checkpoint, args.dataset,
                            preserve_ratio=1. if args.job == 'export' else args.preserve_ratio,
                            n_data_worker=args.n_worker, batch_size=args.data_bsize,
                            args=args, export_model=args.job == 'export', use_new_input=args.use_new_input, text_writer=text_writer)

    text_writer.write('Environment created\n')
    if args.job == 'train':
        
        text_writer.write('Training the agent\n')
        nb_states = env.layer_embedding.shape[1]
        nb_actions = 1  # just 1 action here

        args.rmsize = args.rmsize * len(env.prunable_idx)  # for each layer
        print('** Actual replay buffer size: {}'.format(args.rmsize))
        text_writer.write('** Actual replay buffer size: {}\n'.format(args.rmsize))

        agent = DDPG(nb_states, nb_actions, args, text_writer)
        text_writer.write('Agent Created\n')

        text_writer.write('Training the agent\n')
        train(args.train_episode, agent, env, args.output)
    
    
    elif args.job == 'export':
        text_writer.write('Exporting the model\n')
        export_model(env, args)
        text_writer.write('Model Exported\n')
    else:
        raise RuntimeError('Undefined job {}'.format(args.job))
    

    end_time = time.time()

    text_writer.write("Time taken: {} seconds".format(end_time - start_time))
    print("Time taken: {} seconds".format(end_time - start_time))
    
    text_writer.close()

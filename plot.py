import re
import matplotlib.pyplot as plt
import numpy as np


file_no = 2567
pr = 0.7
lb = 0.5
rb = 1.0



def parse_log_file(file_path):
    episode_rewards = []
    accuracies = []
    episodes = []

    with open(file_path, 'r') as file:
        for line in file.readlines():
            match = re.search(r'#(\d+): episode_reward:([-+]?\d*\.\d+|\d+) acc: ([-+]?\d*\.\d+|\d+)', line)

            if match:
                episodes.append(int(match.group(1)))
                episode_rewards.append(float(match.group(2)))
                accuracies.append(float(match.group(3)))

    return episodes, episode_rewards, accuracies

import matplotlib.colors as colors

def plot_data(episodes, episode_rewards, accuracies):
    accuracies_clipped = np.clip(accuracies, a_min=None, a_max=5000)
    
    plt.figure(figsize=(12, 5))
    plt.suptitle('(Preserve Ratio: {}, Lower Bound: {}, Upper Bound: {})'.format(pr, lb, rb), fontsize=16)
    
    plt.subplot(131)
    plt.plot(episodes, episode_rewards)
    plt.title('Episode reward over time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(132)
    plt.plot(episodes, accuracies)
    plt.title('Perplexity over time')
    plt.xlabel("Episode")
    plt.ylabel('Perplexity')

    plt.subplot(133)
    n, bins, patches = plt.hist(accuracies_clipped[:100], bins=20)
    plt.title('Histogram of first 50 perplexities')
    plt.xlabel('Perplexity')
    plt.ylabel('Frequency')

    # Color code by height
    fracs = n / n.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    # Add frequency numbers on top of each rectangle
    for i in range(len(patches)):
        plt.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height(), str(int(n[i])), ha='center', va='bottom')

    plt.tight_layout()
    print(file_no, pr, lb, rb)
    plt.savefig(f'./plots/histograms/{file_no}_preserve_ratio_{pr}_lbound_{lb}_rbound_{rb}_cased_reward.png')
    plt.show()


# Use the functions
file_path = f'results/AMC_search_test_2567.out'
# file_path = "AMC_search_LLama.out"
episodes, episode_rewards, accuracies = parse_log_file(file_path)
plot_data(episodes, episode_rewards, accuracies)
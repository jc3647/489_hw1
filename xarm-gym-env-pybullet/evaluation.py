import matplotlib.pyplot as plt
import numpy as np

# Function to calculate moving average
def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

def process_file(filename):
    total_rewards = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            reward_part = parts[-1].strip().split(':')
            reward = float(reward_part[-1].strip())
            total_rewards.append(reward)
    return total_rewards

# Filenames
filename1 = "episodeInfoTamerSophieStep0.15N10.txt"
filename2 = "actuallysophie.txt"

# Process data from both files
rewards1 = process_file(filename1)
rewards2 = process_file(filename2)

# Invert the rewards
inverted_rewards1 = [-reward for reward in rewards1][:250]
inverted_rewards2 = [-reward for reward in rewards2][:250]

# Calculate moving averages
window_size = 10
smoothed_rewards1 = moving_average(inverted_rewards1, window_size)
smoothed_rewards2 = moving_average(inverted_rewards2, window_size)

# Plot the rewards
plt.plot(smoothed_rewards1, label='Interactive Q-Learning')
plt.plot(smoothed_rewards2, label='Interactive Q-Learning + TAMER')
plt.xlabel('Episodes')
plt.ylabel('Smoothed Loss (Inverted Cumulative Reward)')
plt.title('Smoothed Loss Function Comparison')
plt.legend()
plt.show()

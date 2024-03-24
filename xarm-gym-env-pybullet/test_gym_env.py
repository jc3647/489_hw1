import gym
import register_xarm_gym
from spawn_goals import main
import numpy as np

env = gym.make('DiscreteXArm7-v0')
nbins = 10

main(p=env.simulation.bullet_client)
for episode in range(10):  # Run 10 episodes
    observation = env.reset()
    for t in range(100):  # Limit the number of steps per episode
        action = 6 # env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(f"Episode finished after {t+1} timesteps")
            break

def create_bins():
    """
    create bins to discretize the continuous observable state space
    0 -> joint 1
    1 -> joint 2
    2 -> joint 3
    3 -> joint 4
    4 -> joint 5
    5 -> joint 6
    6 -> joint 7
    7 -> gripper x
    8 -> gripper y
    9 -> gripper z
    """
    bins = np.zeros((8, nbins))
    bins[0] = np.linspace(-6.283185307179586, 6.283185307179586, nbins)
    bins[1] = np.linspace(-2.059, 2.0944, nbins)
    bins[2] = np.linspace(-6.283185307179586, 6.283185307179586, nbins)
    bins[3] = np.linspace(-0.19198, 3.927, nbins)
    bins[4] = np.linspace(-6.283185307179586, 6.283185307179586, nbins)
    bins[5] = np.linspace(-1.69297, 3.141592653589793, nbins)
    bins[6] = np.linspace(-6.283185307179586, 6.283185307179586, nbins)
    bins[7] = np.linspace(-0.897, 0.496, nbins)
    bins[8] = np.linspace(-1.198, 0.211, nbins)
    bins[9] = np.linspace(0.937, 1.735, nbins)

    return bins

def get_state_as_string(state):
    """
    encoding the state into string as dict
    """
    string_state = ''
    for e in state:
        string_state += str(int(e)).zfill(2)
    return string_state

def get_all_states_as_string():
    states = []
    for i in range (nbins+1):
        for j in range (nbins+1):
            for k in range(nbins+1):
                for l in range(nbins+1):
                    a=str(i).zfill(2)+str(j).zfill(2)+str(k).zfill(2)+str(l).zfill(2)
                    states.append(a)
    return states

def init_Q_table():
    Q = {}
    all_states = get_all_states_as_string()
    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0
    return Q

def play_one_game(bins, Q, eps=0.5):
    observation = env.reset()
    done = False
    


input()
env.close()


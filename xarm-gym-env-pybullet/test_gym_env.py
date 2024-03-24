import gym
import register_xarm_gym
from spawn_goals import main
import numpy as np

env = gym.make('DiscreteXArm7-v0')
main(p=env.simulation.bullet_client)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

class SophiesKitchenRL():
    def __init__(self, episodes=1000, epsilon=0.2, alpha=0.1, gamma=0.99, N=1000, num_actions=6):
        self.env = env
        self.N = N
        self.num_actions = num_actions
        self.episodes = episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.bounds = {
            'x': (-0.897, 0.496),
            'y': (-1.198, 0.211),
            'z': (0.937, 1.735),
        }

        self.run()

    def discretize_state(self, state):
        discretized_state = []
        for i, key in enumerate(self.bounds.keys()):
            min_bound, max_bound = self.bounds[key]
            scaling = (state[i] - min_bound) / (max_bound - min_bound)
            discretized_state.append(int(scaling * self.N))
        return tuple(discretized_state)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            if self.q_table.get(state) is None:
                # self.q_table[state] = np.zeros(self.num_actions)
                self.q_table[state] = np.random.uniform(low=-1, high=1, size=(self.num_actions))
            q_values = self.q_table[state]
            action_probabilities = softmax(q_values)
            action = np.random.choice(self.num_actions, p=action_probabilities)
            return action
            # return np.argmax(self.q_table[state])
        
    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        if self.q_table.get(discrete_state) is None:
            # self.q_table[discrete_state] = np.zeros(self.num_actions)
            self.q_table[discrete_state] = np.random.uniform(low=-1, high=1, size=(self.num_actions))
        if self.q_table.get(discrete_next_state) is None:
            # self.q_table[discrete_next_state] = np.zeros(self.num_actions)
            self.q_table[discrete_next_state] = np.random.uniform(low=-1, high=1, size=(self.num_actions))
        max_future_q = np.max(self.q_table[discrete_next_state])
        current_q = self.q_table[discrete_state][action]    
        new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
        self.q_table[discrete_state][action] = new_q

    def run(self):
        for episode in range(self.episodes):
            count = 0
            tmp = self.env.reset()[0]
            self.discretize_state(tmp)
            state = tmp
            done = False
            goal_position = np.array([0.4919206904119892, -0.32300503747796676, 1.1])

            while not done:
                guidance = False
                action = self.env.greedy_action(state, goal_position)
                if np.random.random() < self.epsilon:
                    if np.random.random() < 0.5:
                        action = self.env.greedy_action(state, goal_position)
                        guidance = True
                    else:
                        action = self.choose_action(state)
                else:  
                    action = self.choose_action(state) 

                observation, reward, done, info = self.env.step(action)
                if guidance:
                    reward += abs(0.5*reward)
                # print(observation, reward, done, info)
                self.update_q_table(state, action, reward, observation, self.alpha, self.gamma)
                state = observation
                print("state ", state)

                # if distance between current state and the goal is less than 0.12, then break
                if np.linalg.norm(np.array([state]) - goal_position) < 0.12:
                    print("Goal reached!")
                    print(count)
                    break

                count += 1

        self.env.close()


test = SophiesKitchenRL(episodes=5)

# def max_dict(d): 
#     """
#     looking for the action that gives the maximum value for a given state
#     """
#     max_v = float('-inf')
#     for key, val in d.items():
#         if val > max_v:
#             max_v = val
#             max_key = key
#     return max_key, max_v

# def create_bins():
#     """
#     create bins to discretize the continuous observable state space
#     0 -> joint 1
#     1 -> joint 2
#     2 -> joint 3
#     3 -> joint 4
#     4 -> joint 5
#     5 -> joint 6
#     6 -> joint 7
#     7 -> gripper x
#     8 -> gripper y
#     9 -> gripper z
#     """
#     bins = np.zeros((10, nbins))
#     bins[0] = np.linspace(-6.283185307179586, 6.283185307179586, nbins)
#     bins[1] = np.linspace(-2.059, 2.0944, nbins)
#     bins[2] = np.linspace(-6.283185307179586, 6.283185307179586, nbins)
#     bins[3] = np.linspace(-0.19198, 3.927, nbins)
#     bins[4] = np.linspace(-6.283185307179586, 6.283185307179586, nbins)
#     bins[5] = np.linspace(-1.69297, 3.141592653589793, nbins)
#     bins[6] = np.linspace(-6.283185307179586, 6.283185307179586, nbins)
#     bins[7] = np.linspace(-0.897, 0.496, nbins)
#     bins[8] = np.linspace(-1.198, 0.211, nbins)
#     bins[9] = np.linspace(0.937, 1.735, nbins)

#     return bins

# def get_state_as_string(state):
#     """
#     encoding the state into string as dict
#     """
#     string_state = ''
#     for e in state:
#         string_state += str(int(e)).zfill(2)
#     return string_state

# def get_all_states_as_string():
#     states = []
#     for i in range (nbins+1):
#         for j in range (nbins+1):
#             for k in range(nbins+1):
#                 for l in range(nbins+1):
#                     a=str(i).zfill(2)+str(j).zfill(2)+str(k).zfill(2)+str(l).zfill(2)
#                     states.append(a)
#     return states

# def init_Q_table():
#     Q = {}
#     all_states = get_all_states_as_string()
#     for state in all_states:
#         Q[state] = {}
#         for action in range(env.action_space.n):
#             Q[state][action] = 0
#     return Q

# def play_one_game(bins, Q, eps=0.5):
#     observation = env.reset()
#     done = False
#     state = get_state_as_string(create_bins(observation, bins))
#     cnt = 0
#     total_reward = 0

#     while not done:
#         cnt += 1
#         if np.random.uniform() < eps:
#             action = env.action_space.sample()
#         else:
#             action = max_dict(Q[state])[0]

#         observation, reward, done, info = env.step(action)

#         total_reward += reward

#         next_state = get_state_as_string(create_bins(observation, bins))
#         a1, max_q_s1a1 = max_dict(Q[next_state])
#         Q[state][action] += 0.1 * (reward + 0.99 * max_q_s1a1 - Q[state][action])
#         state, action = next_state, a1

#     return total_reward, cnt

# def play_many_games(bins, N=100):
#     Q = init_Q_table()
#     length = []
#     reward = []
#     for n in range(N):
#         eps = 1.0 / np.sqrt(n + 1)
#         episode_reward, episode_length = play_one_game(bins, Q, eps)
#         if n % 100 == 0:
#             print(n, '%.4f' % eps, episode_reward)
#         length.append(episode_length)
#         reward.append(episode_reward)
#     # env.close()
#     return length, reward

# main(p=env.simulation.bullet_client)
# for episode in range(10):  # Run 10 episodes
#     play_many_games(create_bins())

# #     observation = env.reset()
# #     for t in range(100):  # Limit the number of steps per episode
# #         action = 6 # env.action_space.sample()
# #         observation, reward, done, info = env.step(action)
# #         if done:
# #             print(f"Episode finished after {t+1} timesteps")
# #             break
# # play_one_game(create_bins(), init_Q_table())
#         # observation, reward, done, info = env.step(action)
#         # next_state = get_state_as_string(create_bins(observation, bins))
#         # next_max = max(Q[next_state], key=Q[next_state].get)
#         # Q[state][action] = reward + 0.99 * Q[next_state][next_max]
#         # state = next_state



# input()
# env.close()


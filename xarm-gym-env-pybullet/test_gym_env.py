import gym
import register_xarm_gym
from spawn_goals import main
import numpy as np
import time
import json

env = gym.make('DiscreteXArm7-v0')
main(p=env.simulation.bullet_client)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

class SophiesKitchenTamerRL():
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

        # TAMER
        self.historical_feedback = {}

    # TAMER
    def update_feedback_history(self, state, action, feedback):
        # TODO
        state = self.discretize_state(state)
        if self.historical_feedback.get(state) is None:
            self.historical_feedback[state] = np.random.uniform(low=-0.2, high=0.2, size=(self.num_actions))
        self.historical_feedback[state][action] = feedback

    def estimate_feedback_for_action(self, state):
        # TODO
        pass

    # Incorporate TAMER to rest of the pipeline

    def discretize_state(self, state):
        discretized_state = []
        for i, key in enumerate(self.bounds.keys()):
            min_bound, max_bound = self.bounds[key]
            scaling = (state[i] - min_bound) / (max_bound - min_bound)
            discretized_state.append(int(scaling * self.N))
        return tuple(discretized_state)

    def choose_action(self, state, testing=False, policy=None):

        state = self.discretize_state(state)

        # test out extracted policy
        if testing:
            if state not in policy.keys():
                return np.random.choice(self.num_actions)
            return policy[state]

        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            if self.q_table.get(state) is None:
                self.q_table[state] = np.random.uniform(low=-0.2, high=0.2, size=(self.num_actions))
            q_values = self.q_table[state] # goal_state - current_state
            action_probabilities = softmax(q_values)
            action = np.random.choice(self.num_actions, p=action_probabilities)
            return action
            # return np.argmax(self.q_table[state])
        
    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        if self.q_table.get(discrete_state) is None:
            self.q_table[discrete_state] = np.random.uniform(low=-0.2, high=0.2, size=(self.num_actions))
        if self.q_table.get(discrete_next_state) is None:
            self.q_table[discrete_next_state] = np.random.uniform(low=-0.2, high=0.2, size=(self.num_actions))
        max_future_q = np.max(self.q_table[discrete_next_state])
        current_q = self.q_table[discrete_state][action]    
        new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
        self.q_table[discrete_state][action] = new_q

    def extract_policy(self, filename):
        policy = {}
        for state, action_values in self.q_table.items():
            best_action = np.argmax(action_values)
            policy[state] = best_action

        np.save(filename, policy)

    def train(self, goal_position):

        origin = self.env.get_current_gripper_pose()

        for episode in range(self.episodes):

            count = 0
            tmp = self.env.reset()[0]
            self.discretize_state(tmp)
            state = tmp
            done = False

            while not done:
                guidance = False

                # action = self.env.choose_action(state)

                if np.random.random() < self.epsilon:
                        action = self.env.greedy_action(state, goal_position)
                        guidance = True
                else:  
                    action = self.choose_action(state) 

                observation, reward, done, info = self.env.step(action)
                if guidance:
                    reward += abs(0.5*reward)

                # print(observation, reward, done, info)
                    
                self.update_q_table(state, action, reward, observation, self.alpha, self.gamma)
                state = observation

                # if distance between current state and the goal is less than 0.12, then break
                if np.linalg.norm(np.array([state]) - goal_position) < 0.12:

                    with open("episodeInfo2.txt", "a") as file:
                        message = f"Episode {episode}, Goal reached at timestep: {count}, States explored: {len(self.q_table)}\n"
                        print(message)
                        file.write(message)
                    
                    self.env.set_gripper_position(origin)
                    time.sleep(1)
                    break

                count += 1

        self.extract_policy('greedyHumanPolicy1')

        self.env.close()

    def test(self, policy, goal_position):
        origin = self.env.get_current_gripper_pose()
        for episode in range(1):
            count = 0
            tmp = self.env.reset()[0]
            self.discretize_state(tmp)
            state = tmp
            done = False

            while not done:
                action = self.choose_action(state, testing=True, policy=policy)
                observation, reward, done, info = self.env.step(action)
                state = observation

                if np.linalg.norm(np.array([state]) - goal_position) < 0.12:
                    with open("trainedInfo1.txt", "a") as file:
                        message = f"Episode {episode}, Goal reached at timestep: {count}, States explored: {len(self.q_table)}\n"
                        print(message)
                        file.write(message)
                    
                    self.env.set_gripper_position(origin)
                    time.sleep(1)
                    break

                count += 1

        self.env.close()



test = SophiesKitchenTamerRL(episodes=3)
test.train(np.array([0.4919206904119892, -0.32300503747796676, 1.1]))
# test.test(np.array([0.4919206904119892, -0.32300503747796676, 1.1]))
        

read_dictionary = np.load('greedyHumanPolicy1.npy',allow_pickle='TRUE').item()
print(read_dictionary)

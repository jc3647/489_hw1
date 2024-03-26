import gym
import register_xarm_gym
from spawn_goals import main, delete_model
import numpy as np
import time
import json

env = gym.make('DiscreteXArm7-v0')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

class SophiesKitchenTamerRL():
    def __init__(self, episodes=1000, epsilon=0.2, alpha=0.1, gamma=0.99, N=100, num_actions=6):
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
            'y': (-1.3, 0.211),
            'z': (-1.6, 1.735),
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
            # print("state[i]: ", state[i], min_bound)
            scaling = (state[i] - min_bound) / (max_bound - min_bound)
            # print("scaling: ", scaling, i)
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
            q_values = self.q_table[state]
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

    def train(self):

        origin = self.env.get_current_gripper_pose()

        for episode in range(self.episodes):

            # save a new policy every 50 episodes
            if episode % 50 == 0:
                self.extract_policy(f'greedyHumanPolicy{episode}')

            if episode == 5:
                self.extract_policy('greedyHumanPolicy5')


            count = 0
            tmp = self.env.reset()[0]
            state = np.array(list(tmp))
            done = False

            random_env = np.random.choice([1, 2, 3])
            new_goals, new_goals_positions  = main(p=self.env.simulation.bullet_client, env=random_env)
            random_goal = np.random.choice([0, 1, 2])
            goal_state = np.array(new_goals_positions[random_goal])
            decoy_locations = new_goals_positions[4:]

            # print("random goal state: ", goal_state)

            self.env.update_goal_state(goal_state)

            while not done:

                # think of goal_state as the origin, and transformed_state to be relative to goal_state
                transformed_state = state - goal_state
                # print("transformed state: ", transformed_state)
                guidance = False

                # for strictly Q-learning
                # action = self.choose_action(state)
                # for strictly greedy
                # action = self.env.greedy_action(state)
                # guidance = True

                if np.random.random() < self.epsilon:
                        action = self.env.greedy_action(state)
                        guidance = True
                else:  
                    action = self.choose_action(transformed_state) 
                observation, reward, done, info = self.env.step(action)
                if guidance:
                    reward += abs(0.5*reward)

                # print(observation, reward, done, info)
                
                transformed_observation = observation - goal_state
                    
                self.update_q_table(transformed_state, action, reward, transformed_observation, self.alpha, self.gamma)
                state = observation

                # got close enough to end goal
                # print("current state: ", state, "goal state: ", goal_state)
                # print("distance to goal: ", np.linalg.norm(np.array([state]) - goal_state))
                if np.linalg.norm(np.array([state]) - goal_state) < 0.05:
                    with open("episodeInfo2.txt", "a") as file:
                        message = f"Episode {episode}, Goal reached at timestep: {count}, States explored: {len(self.q_table)}\n"
                        print(message)
                        file.write(message)
                    for obj in new_goals:
                        delete_model(self.env.simulation.bullet_client, obj[0])
                    self.env.set_gripper_position(origin)
                    time.sleep(1)
                    break

                count += 1

        self.extract_policy('greedyHumanPolicy2')

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



test = SophiesKitchenTamerRL(episodes=1000000000)
test.train()
# test.test(np.array([0.4919206904119892, -0.32300503747796676, 1.1]))
        
# read_dictionary = np.load('greedyHumanPolicy1.npy',allow_pickle='TRUE').item()
# print(read_dictionary)

from gym import spaces
import numpy as np
import gym
from env.xarm_sim import XArm7Sim
import pybullet as p
import pybullet_data as pd

class DiscreteXArm7GymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, step_size=0.1):
        super(DiscreteXArm7GymEnv, self).__init__()

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pd.getDataPath())

        timeStep = 1. / 60.
        p.setTimeStep(timeStep)
        p.setGravity(0, 0, -9.8)
        self.simulation = XArm7Sim(p)

        self.step_size = step_size  # Movement step size for the gripper

        # Define a discrete action space: 7 actions for +/- movement in each axis and no-op
        self.action_space = spaces.Discrete(7)

        # Observation space representing the 3D position of the gripper
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.goal_state = None

    def update_goal_state(self, goal_state):
        self.goal_state = goal_state

    def step(self, action):

        # Map the action to a change in gripper position
        delta = np.zeros(3)
        if action == 0:
            delta[0] += self.step_size  # Move +X
        elif action == 1:
            delta[0] -= self.step_size  # Move -X
        elif action == 2:
            delta[1] += self.step_size  # Move +Y
        elif action == 3:
            delta[1] -= self.step_size  # Move -Y
        elif action == 4:
            delta[2] += self.step_size  # Move +Z
        elif action == 5:
            delta[2] -= self.step_size  # Move -Z
        # action == 0 is no-op

        current_pose = self.simulation.get_current_gripper_pose()
        new_pose = current_pose + delta
        self.simulation.set_gripper_position(new_pose)

        # Implement reward calculation, check if the episode is done, etc.

        # reward is distance to goal
        distance = np.sqrt((self.goal_state[0] - new_pose[0])**2 +
                           (self.goal_state[1] - new_pose[1])**2 +
                            (self.goal_state[2] - new_pose[2])**2)
        
        reward = -distance
        done = False
        info = {}

        return self.get_current_gripper_pose(), reward, done, info

    def greedy_action(self, state):
        lst = []
        for action in range(6):
            delta = np.zeros(3)
            if action == 0:
                delta[0] += self.step_size
            elif action == 1:
                delta[0] -= self.step_size
            elif action == 2:
                delta[1] += self.step_size
            elif action == 3:
                delta[1] -= self.step_size
            elif action == 4:
                delta[2] += self.step_size
            elif action == 5:
                delta[2] -= self.step_size
            new_pose = state + delta
            distance = np.linalg.norm(self.goal_state - new_pose)
            lst.append((distance, action))

        return min(lst)[1]


    def reset(self):
        # Reset simulation and return initial observation
        self.simulation.reset()
        # Implement reward calculation, check if the episode is done, etc.
        reward = 0
        done = False
        info = {}

        return self.get_current_gripper_pose(), reward, done, info

    def render(self, mode='human', close=False):
        # Optional: Implement to visualize the simulation
        self.simulation.bullet_client.stepSimulation()

    def get_current_gripper_pose(self):
        # Implement to return the current 3D position of the gripper
        return self.simulation.get_current_gripper_pose()

    def set_gripper_position(self, position):
        # Implement to move the gripper to the specified position using your simulation
        return self.simulation.set_gripper_position(position)

import gym
import register_xarm_gym
from spawn_goals import main

env = gym.make('DiscreteXArm7-v0')
main(p=env.simulation.bullet_client)
for episode in range(10):  # Run 10 episodes
    observation = env.reset()
    for t in range(100):  # Limit the number of steps per episode
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print(f"Episode finished after {t+1} timesteps")
            break
input()
env.close()


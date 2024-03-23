from gym.envs.registration import register

register(
    id='DiscreteXArm7-v0',
    entry_point='xarm_gym_env:DiscreteXArm7GymEnv',
)

**CPSC 489: Robot Learning - Homework 1: RL and IRL**

![ezgif-4-7625396c9c](https://github.com/jc3647/489_hw1/assets/65755432/57b187d0-04d4-49fc-a54f-746f89f8b20e)


For my implementations, I focused on reinforcement learning and wrote code for Interactive Q-Learning (Thomaz & Breazeal 2006) and TAMER (Knox & Stone 2009). Below are the algorithms:

![image](https://github.com/jc3647/489_hw1/assets/65755432/7e4d808a-02ab-46b6-af19-df102c713f2f)

![image](https://github.com/jc3647/489_hw1/assets/65755432/bedfd854-8b05-4ea0-8811-5f4de68a9e13)

The three behaviors I decided on are:

1. Go to target block position
2. Go to target decoy position
3. Go to target block while avoiding a target decoy

In terms of my RL setup, I decided to discretize the observation space. I tested for the limits that the arm can reach and used these as the bounds when discretizing the space into individual bins. To move the xArm, I take step sizes of 0.1 in cartesian space.

To calculate the reward when updating the q-values in the observation space, I used the negative distance from the goal position to the current end effector position (the closer the distance, the larger the reward). For Interactive Q-Learning, I had a greedy agent that provided "guidance" (the next optimal step to take) with some epsilon percent chance on each step, and this step is given an extra weighted reward. For TAMER, I again had a greedy agent, this time giving reward to a step that would shorten the distance between the current end effector position and the goal position.

For further results, refer to Report.pdf found within this repositiory.

_Note: I used Liam's PyBullet environment that he published on EdStem (https://edstem.org/us/courses/56634/discussion/4603688)_

Thomaz, A. L., & Breazeal, C. (2006, July). Reinforcement learning with human teachers: Evidence of feedback and guidance with implications for learning performance. In Aaai (Vol. 6, pp. 1000-1005).

Knox, W. B., & Stone, P. (2009, September). Interactively shaping agents via human reinforcement: The TAMER framework. In Proceedings of the fifth international conference on Knowledge capture (pp. 9-16).

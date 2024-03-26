**CPSC 489: Robot Learning - Homework 1: RL and IRL**

![image](https://github.com/jc3647/489_hw1/assets/65755432/f32035b7-c373-4e08-8fd9-8f2852942d4b)

For my implementations, I focused on reinforcement learning and wrote code for Interactive Q-Learning (Thomaz & Breazeal 2006) and TAMER (Knox & Stone 2009). Below are the algorithms:

![image](https://github.com/jc3647/489_hw1/assets/65755432/7e4d808a-02ab-46b6-af19-df102c713f2f)

![image](https://github.com/jc3647/489_hw1/assets/65755432/bedfd854-8b05-4ea0-8811-5f4de68a9e13)

The three behaviors I decided on are:

1. Go to target block position
2. Go to target decoy position
3. Go to target block while avoiding a target decoy

In terms of my RL setup, I decided to discretize the observation space. I tested for the limits that the arm can reach and used these as the bounds when discretizing the space into individual bins. To move the xArm, I take step sizes of 0.1 in cartesian space.



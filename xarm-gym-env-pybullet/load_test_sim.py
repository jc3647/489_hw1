import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
# import pybullet_robots.xarm.xarm_sim as xarm_sim
import env.xarm_sim as xarm_sim

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

timeStep=1./60.
p.setTimeStep(timeStep)
p.setGravity(0,0,-9.8)

"""Load joint angles from a specified file."""
joint_angles_list = []
file_path = "/home/lab/489_hw1/planning/joint_space_full_traj/shortest_path_env_0_goal_0_traj_1.txt"
with open(file_path, 'r') as file:
    lines = file.readlines()
    
    joint_angles = [[float(angle) for angle in line.strip()[1:-1].split(',')] for line in lines]
 
xarm = xarm_sim.XArm7Sim(p,[0,0,0])
xarm.execute_traj(joint_angles)
while (1):
	
	p.stepSimulation()
	time.sleep(timeStep)
	
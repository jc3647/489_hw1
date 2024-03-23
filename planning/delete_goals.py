import sys
import yaml
import argparse
import os




with open('../config/goals.yaml','r') as file:
    data1 = yaml.safe_load(file)



# parser = argparse.ArgumentParser()
# parser.add_argument("-env", "--environment", dest = "env", default = 1, help="Environment number", type=int)
# parser.add_argument("-g", "--goal", dest = "g", default = 1, help="Goal number", type=int)
# parser.add_argument("-traj", "--trajectory",dest ="traj", default = 1, help="Trajectory number", type=int)
# parser.add_argument("-wpt", "--waypoint",dest = "wpt", default = 1, help="Waypoint number", type=int)

# args = parser.parse_args()

# i = args.env - 1 # 1~3
# j = args.g - 1 # 1~8
# m = args.traj # 1~3
# n = args.wpt # 0~5

#print(data1['env1']['end1'])
#os.system('python3 delete_entity.py -entity box_start -file ~/.gazebo/models/box/model.sdf')
os.system('python3 delete_entity.py -entity box -file ../config/box/model.sdf')
os.system('python3 delete_entity.py -entity box_2 -file ../config/box/model.sdf')
os.system('python3 delete_entity.py -entity box_3 -file ../config/box/model.sdf')
os.system('python3 delete_entity.py -entity box_4 -file ../config/box/model.sdf')
os.system('python3 delete_entity.py -entity decoy1 -file ../config/decoys/model1.sdf')
os.system('python3 delete_entity.py -entity decoy2 -file ../config/decoys/model2.sdf')
os.system('python3 delete_entity.py -entity decoy3 -file ../config/decoys/model3.sdf')
os.system('python3 delete_entity.py -entity decoy4 -file ../config/decoys/model4.sdf')
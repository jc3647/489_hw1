import sys
import yaml
import argparse
import os


with open('../config/pickup_position.yaml','r') as file:
    data = yaml.safe_load(file)

with open('../config/goals.yaml','r') as file:
    data1 = yaml.safe_load(file)



parser = argparse.ArgumentParser()
parser.add_argument("-env", "--environment", dest = "env", default = 1, help="Environment number", type=int)
#parser.add_argument("-g", "--goal", dest = "g", default = 1, help="Goal number", type=int)
#parser.add_argument("-traj", "--trajectory",dest ="traj", default = 1, help="Trajectory number", type=int)
# parser.add_argument("-wpt", "--waypoint",dest = "wpt", default = 1, help="Waypoint number", type=int)

args = parser.parse_args()

i = args.env  # 1~3
#j = args.g  # 1~8
#m = args.traj # 1~3
# n = args.wpt # 0~5

#print(data1['env1']['end1'])
# os.system('python3 spawn_entity.py -entity box_start -x '+str(data['circle']['block_1'][1]-0.2)+' -y '+str(data['circle']['block_1'][0]-0.5)+
#     ' -z '+str(data['circle']['block_1'][2])+' -file ~/.gazebo/models/box_start/model.sdf') # x and y converted; x y positions adjusted based on the arm position in rviz
#print(data1['env'+str(i)]['end1'])
# goal placing is inverted wrt y axis so the spawning is different from normal; switch y and z between the symetric boxes
os.system('python3 spawn_entity.py -entity box -x '+str(data1['env'+str(i)]['end1'][1]-0.2)+' -y '+str(data1['env'+str(i)]['end4'][0]-0.5)+
        ' -z '+str(data1['env'+str(i)]['end4'][2])+' -file ../config/box/model.sdf')
os.system('python3 spawn_entity.py -entity box_2 -x '+str(data1['env'+str(i)]['end2'][1]-0.2)+' -y '+str(data1['env'+str(i)]['end3'][0]-0.5)+
          ' -z '+str(data1['env'+str(i)]['end3'][2])+' -file ../config/box_2/model.sdf')
os.system('python3 spawn_entity.py -entity box_3 -x '+str(data1['env'+str(i)]['end3'][1]-0.2)+' -y '+str(data1['env'+str(i)]['end2'][0]-0.5)+
          ' -z '+str(data1['env'+str(i)]['end2'][2])+' -file ../config/box_3/model.sdf')
os.system('python3 spawn_entity.py -entity box_4 -x '+str(data1['env'+str(i)]['end4'][1]-0.2)+' -y '+str(data1['env'+str(i)]['end1'][0]-0.5)+
          ' -z '+str(data1['env'+str(i)]['end1'][2])+' -file ../config/box_4/model.sdf')
os.system('python3 spawn_entity.py -entity decoy1 -x '+str(data1['env'+str(i)]['end5'][1]-0.2)+' -y '+str(data1['env'+str(i)]['end8'][0]-0.5)+
          ' -z '+str(data1['env'+str(i)]['end8'][2])+' -file ../config/decoys/model1.sdf')
os.system('python3 spawn_entity.py -entity decoy2 -x '+str(data1['env'+str(i)]['end6'][1]-0.2)+' -y '+str(data1['env'+str(i)]['end7'][0]-0.5)+
          ' -z '+str(data1['env'+str(i)]['end7'][2])+' -file ../config/decoys/model2.sdf')
os.system('python3 spawn_entity.py -entity decoy3 -x '+str(data1['env'+str(i)]['end7'][1]-0.2)+' -y '+str(data1['env'+str(i)]['end6'][0]-0.5)+
          ' -z '+str(data1['env'+str(i)]['end6'][2])+' -file ../config/decoys/model3.sdf')
os.system('python3 spawn_entity.py -entity decoy4 -x '+str(data1['env'+str(i)]['end8'][1]-0.2)+' -y '+str(data1['env'+str(i)]['end5'][0]-0.5)+
          ' -z '+str(data1['env'+str(i)]['end5'][2])+' -file ../config/decoys/model4.sdf')


import os

for env in range(3):
    for goal in range(4):
        for traj in range(3):
            os.system(f"python3 eef_traj.py -env {env+1} -g {goal+1} -traj {traj+1}")
import time
import numpy as np
import math
import os

import pybullet

useNullSpace = 0
useDynamics = 1
useIKFast = 0  # ikfast doesn't get solutions and is actually slower than pybullet IK (300us versus 150us), probably a configuration issue
if useIKFast:
    import ikfastpy

ikSolver = 0
xarmEndEffectorIndex = 7
xarmNumDofs = 7

DEMO_DIR = '/home/lab/489_hw1/planning/joint_space_full_traj'
URDF_PATH = '/home/lab/489_hw1/xarm-gym-env-pybullet/env/xarm7_pd.urdf'

# DEMO_DIR = '/home/liam/Documents/phd/classes/Robot_learning/planning/joint_space_full_traj/'
# URDF_PATH = '/home/liam/dev_ws/src/TAMER/tamer/env/xarm7_pd.urdf'

ll = [-17] * xarmNumDofs
# upper limits for null space (todo: set them to proper range)
ul = [17] * xarmNumDofs
# joint ranges for null space (todo: set them to proper range)
jr = [17] * xarmNumDofs
# restposes for null space
jointPositions = [0, 0, 0, 0, 0, 0, 0]
zeroPosition = jointPositions
rp = jointPositions
jointPoses = jointPositions


class XArm7Sim(object):
    def __init__(self, bullet_client, trans_offset=(-0.2, -0.5, 1.021), rot_offset=(0, 0, 0.7071, 0.7071)):
        global xarmEndEffectorIndex, xarmNumDofs
        if useIKFast:
            # Initialize kinematics for UR5 robot arm
            self.xarm_kin = ikfastpy.PyKinematics()
            self.n_joints = self.xarm_kin.getDOF()
            print("numJoints IKFast=", self.n_joints)

        self.bullet_client = bullet_client
        self.offset = np.array(trans_offset)
        self.rot_offset = np.array(rot_offset)
        self.jointPoses = [0] * xarmNumDofs
        # print("offset=",offset)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.bullet_client.setRealTimeSimulation(1)
        legos = []
        orn = np.array((0,0,0,0)) +self.rot_offset
        self.xarm = self.bullet_client.loadURDF(URDF_PATH, np.array([0, 0, 0]) + self.offset, orn, useFixedBase=True,
                                                flags=flags)
        xarmEndEffectorIndex = self.bullet_client.getNumJoints(self.xarm) - 1
        xarmNumDofs = 7
        print(self.bullet_client.getNumJoints(self.xarm))
        print()
        print()
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.xarm)):
            self.bullet_client.changeDynamics(self.xarm, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.xarm, j)

            jointName = info[1]
            jointType = info[2]
            # if (jointType == self.bullet_client.JOINT_PRISMATIC):
            #
            #   self.bullet_client.resetJointState(self.xarm, j, jointPositions[index])
            #   index=index+1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.xarm, j, jointPositions[index])
                index = index + 1
        self.t = 0.

    def reset(self):
        for i in range(xarmNumDofs):
            self.bullet_client.setJointMotorControl2(self.xarm, i + 1, self.bullet_client.POSITION_CONTROL,
                                                     zeroPosition[i], force=5 * 240.)

    def set_gripper_position(self, pose):
        jointPoses = self.get_ik(pose)
        self.execute_traj([jointPoses])

    def get_ik(self, pose):
        orn = [1, 0, 0, 0]
        return self.bullet_client.calculateInverseKinematics(self.xarm, xarmEndEffectorIndex, pose, orn,
                                                             maxNumIterations=50)

    def get_fk(self, jointPoses):
        for i in range(xarmNumDofs):
            print(jointPoses)
            self.bullet_client.setJointMotorControl2(self.xarm, i + 1, self.bullet_client.POSITION_CONTROL,
                                                     jointPoses[i], force=5 * 240.)
        self.bullet_client.stepSimulation()
        ls = self.bullet_client.getLinkState(self.xarm, xarmEndEffectorIndex, computeForwardKinematics=True)
        linkComPos = np.array(ls[0])
        return linkComPos

    def execute_traj(self, joint_poses):
        for pos in joint_poses:
            t = self.t
            self.t += 1. / 60.
            orn = [1, 0, 0, 0]

            print(pos)
            if useNullSpace:
                jointPoses = self.bullet_client.calculateInverseKinematics(self.xarm, xarmEndEffectorIndex, pos, orn,
                                                                           lowerLimits=ll,
                                                                           upperLimits=ul, jointRanges=jr,
                                                                           restPoses=np.array(self.jointPoses).tolist(),
                                                                           residualThreshold=1e-5, maxNumIterations=50)
                self.jointPoses = [0, 0, jointPoses[2], jointPoses[3], jointPoses[4], jointPoses[5]]
            # else:
                # cart = self.get_fk(pos)
                #
                # jointPoses = self.get_ik(
                #     pos)  # self.bullet_client.calculateInverseKinematics(self.xarm,xarmEndEffectorIndex, pos, orn, maxNumIterations=50)
            # print("jointPoses=",jointPoses)
            if useDynamics:
                for i in range(xarmNumDofs):
                    self.bullet_client.setJointMotorControl2(self.xarm, i + 1, self.bullet_client.POSITION_CONTROL,
                                                             pos[i], force=5 * 240.)
            else:
                for i in range(xarmNumDofs):
                    self.bullet_client.setJointMotorControl2(self.xarm, i + 1, pybullet.POSITION_CONTROL, targetPosition=pos[i])
            ls = self.bullet_client.getLinkState(self.xarm, xarmEndEffectorIndex, computeForwardKinematics=True)
            linkComPos = np.array(ls[0])
            linkComOrn = ls[1]
            linkUrdfPos = np.array(ls[4])
            linkUrdfOrn = ls[5]
            # print("linkComPos=",linkComPos)
            # print("linkUrdfOrn=",linkUrdfOrn)
            mat = self.bullet_client.getMatrixFromQuaternion(linkUrdfOrn)
            time.sleep(1 / 60)
            # print("mat=",mat)
            # self.bullet_client.addUserDebugLine(pos, linkUrdfPos, [1,0,0,1],lifeTime=100)
            # diff = linkUrdfPos-np.array(pos)
            # print("diff=",diff)

    def get_current_gripper_pose(self):
        state = self.bullet_client.getLinkState(self.xarm, xarmEndEffectorIndex)
        return state[0]

    def get_demos(self):
        demos = []
        for _, _, files in os.walk(DEMO_DIR):
            demo = []
            for file in files:
                file_path = os.path.join(DEMO_DIR, file)
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                    joint_angles = [[float(angle) for angle in line.strip()[1:-1].split(',')] for line in lines]
                    demos.append(joint_angles)
        self.demos = demos
        return demos

import sys

from xarm_msgs.srv import PlanJoint
from xarm_msgs.srv import PlanExec
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
import yaml
import numpy as np
import sys
import argparse

from std_msgs.msg import String
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from tf2_msgs.msg import TFMessage

import numpy as np

import pdb

class xarmJointPlanningAndRecordingClient(Node):

    def __init__(self):
        super().__init__('xarm_joint_planning_client')
        self.plan_cli = self.create_client(PlanJoint, 'xarm_joint_plan')
        self.exec_cli = self.create_client(PlanExec, 'xarm_exec_plan')
        
        self.target_frame = 'link_eef'
    
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def send_plan_request(self,pose):
        while not self.plan_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.plan_req = PlanJoint.Request()
        self.plan_req.target = pose
        self.plan_future = self.plan_cli.call_async(self.plan_req)
        rclpy.spin_until_future_complete(self, self.plan_future)
        return self.plan_future.result()
    
    def send_exec(self):
        self.get_logger().info('Planning success executing...')
        while not self.exec_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.exec_req = PlanExec.Request()
        self.exec_req.wait = True
        self.exec_future = self.exec_cli.call_async(self.exec_req)
        rclpy.spin_until_future_complete(self, self.exec_future)
        return self.exec_future.result()  

    def plan_and_execute(self,pose):
        while not self.plan_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.plan_req = PlanJoint.Request()
        self.plan_req.target = pose
        self.plan_future = self.plan_cli.call_async(self.plan_req)
        rclpy.spin_until_future_complete(self, self.plan_future)
        res = self.plan_future.result()
        if res.success:
            self.get_logger().info('Planning success executing...')
            while not self.exec_cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.exec_req = PlanExec.Request()
            self.exec_req.wait = True
            self.exec_future = self.exec_cli.call_async(self.exec_req)
            rclpy.spin_until_future_complete(self, self.exec_future)
            return self.exec_future.result()
        else:
            self.get_logger().info('Planning Failed!')
            return False



def main():
    rclpy.init()
    client = xarmJointPlanningAndRecordingClient()

    parser = argparse.ArgumentParser()
    parser.add_argument("-env", "--environment", dest = "env", default = 1, help="Environment number", type=int)
    parser.add_argument("-g", "--goal", dest = "g", default = 1, help="Goal number", type=int)
    parser.add_argument("-traj", "--trajectory",dest ="traj", default = 1, help="Trajectory number", type=int)
    #parser.add_argument("-seg", "--segment",dest = "seg", default = 1, help="Segment", type=int)

    args = parser.parse_args()

    env = args.env - 1 # 1~3
    g = args.g - 1 # 1~8
    traj = args.traj # 1~3
    #n = args.seg # 1~5

    filename_prefix = "./joint_space_full_traj/shortest_path_env_" +str(env)+"_goal_"+str(g)+"_traj_"+str(traj)
    eef_file = filename_prefix + "_eef.txt"

    # whole traj
    #print(i)
    with open(filename_prefix+".txt",'r') as file:
    #data = file.readlines()
        data = file.read().splitlines()
    file.close()
       
    jt_traj = []
    for lines in data:
        jt_traj.append(lines[1:-1])
    #print(jt_traj)

    with open(eef_file, 'w') as file:
        for i in range(len(jt_traj)):
            target_pose = jt_traj[i].split(',')
            target_pose = [float(m) for m in target_pose]
            response = client.plan_and_execute(target_pose)
            print(response)

            from_frame_rel = 'link_eef'
            to_frame_rel = 'world'

            # pdb.set_trace()
            now = rclpy.time.Time()
            trans = client.tf_buffer.lookup_transform(
                        to_frame_rel,
                        from_frame_rel,
                        now)

            pose_now = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
                # trans.transform.rotation.x,
                # trans.transform.rotation.y,
                # trans.transform.rotation.z,
                # trans.transform.rotation.w
            ]

            file.write(str(pose_now)+'\n')


    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from tf2_msgs.msg import TFMessage

import numpy as np


class eefPublisher(Node):

  def __init__(self):
    
    super().__init__('eef_link_listener')
    self.publisher_ = self.create_publisher(String, 'eef_traj', 10) # can create the message you want to publish
    self.declare_parameter('target_frame', 'link_eef')
    self.target_frame = self.get_parameter(
      'target_frame').get_parameter_value().string_value
 
    self.tf_buffer = Buffer()
    self.tf_listener = TransformListener(self.tf_buffer, self)

    self.file = "./eef_trajectory.txt"
    #open(self.file, 'w').close()

    self.sensitivity = 0.0001
    timer_period = 2.0
    self.timer = self.create_timer(timer_period, self.on_timer)

    self.pose_current = [0.0,0.0,0.0,0.0,0.0,0.0,0.0] #7dof
     
  def on_timer(self):
    from_frame_rel = self.target_frame
    to_frame_rel = 'world'
   
    trans = None
     
    try:
      now = rclpy.time.Time()
      trans = self.tf_buffer.lookup_transform(
                  to_frame_rel,
                  from_frame_rel,
                  now)
    except TransformException as ex:
      self.get_logger().info(
        f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
      return
    
    
    pose_now = [trans.transform.translation.x,trans.transform.translation.y,trans.transform.translation.z,
                trans.transform.rotation.x,trans.transform.rotation.y,trans.transform.rotation.z,trans.transform.rotation.w]
    
    # if np.linalg.norm(np.array(self.pose_current)-np.array(pose_now))>self.sensitivity:
    #     self.pose_current = pose_now
    #     self.get_logger().info('Recording eef pose.')
    #     with open(self.file, 'a') as file:
    #         file.write(str(self.pose_current)+'\n')
    # else:
    #   self.get_logger().info('Robot is stationary. Not recording!')
    #self.get_logger().info(str(pose_now))
    msg = String()
    msg.data = str(pose_now)
    self.publisher_.publish(msg)
    self.get_logger().info(msg.data)
    #self.i += 1


def main(args=None):
  rclpy.init(args=args)
  eef_listener_node = eefPublisher()
  try:
    rclpy.spin(eef_listener_node)
  except KeyboardInterrupt:
    pass
  rclpy.shutdown()


if __name__ == '__main__':
    main()
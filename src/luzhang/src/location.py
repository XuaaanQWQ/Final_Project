#!/usr/bin/env python3
import rospy
import tf
import numpy as np
from geometry_msgs.msg import PointStamped, Point

class Orange:
    def __init__(self):
        rospy.init_node("location", anonymous=True)
        rospy.Subscriber("/orange_center", Point, self.orange_callback)
        self.map_pub = rospy.Publisher("/target", PointStamped, queue_size=10)
        
        self.tf_listener = tf.TransformListener()

    def get_robot_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
            x_robot, y_robot = trans[0], trans[1]
            _, _, yaw = tf.transformations.euler_from_quaternion(rot)
            return x_robot, y_robot, yaw
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("no (map → base_link)")
            return None, None, None

    def get_camera_transform(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform("base_link", "front_realsense_lens", rospy.Time(0))
            return trans[0], trans[1], trans[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("no (base_link → front_realsense_lens joint)")
            return 0.1, 0.0, 0.1 

    def orange_callback(self, msg):
        cx, cy, depth = msg.x, msg.y, msg.z

        if depth <= 0 or np.isnan(depth):
            rospy.logwarn(f"no depth")
            return

        x_robot, y_robot, theta_robot = self.get_robot_pose()
        if x_robot is None:
            rospy.logwarn("no pose")
            return

        x_offset, y_offset, z_offset = self.get_camera_transform()

        fx, fy = 600, 600  
        img_center_x, img_center_y = 320, 240  

        X_cam = (cx - img_center_x) * depth / fx
        Y_cam = (cy - img_center_y) * depth / fy
        Z_cam = depth

        X_base = X_cam + x_offset
        Y_base = Y_cam + y_offset

        x_map = x_robot + X_base * np.cos(theta_robot) - Y_base * np.sin(theta_robot)
        y_map = y_robot + X_base * np.sin(theta_robot) + Y_base * np.cos(theta_robot)

        y_map += 2.0  

        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = "map"
        point_msg.point.x = x_map
        point_msg.point.y = y_map
        point_msg.point.z = 0
        self.map_pub.publish(point_msg)

        rospy.loginfo(f"target: (x={x_map:.2f}, y={y_map:.2f})")

if __name__ == "__main__":
    Orange()
    rospy.spin()

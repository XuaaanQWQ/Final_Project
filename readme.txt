# 用到的包如下
import rospy
import cv2
import numpy as np
import tf
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from nav_msgs.msg import Odometry

# 需要注意模型文件的路径
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import cv2

from cv_bridge import CvBridge, CvBridgeError
import rospkg
import rospy
from sensor_msgs.msg import Image

package_path = rospkg.RosPack().get_path('centernet_ros')
sys.path.insert(0, package_path + '/CenterNet/src/lib/')
sys.path.insert(0, package_path + '/CenterNet/src/')

from opts import opts
from detectors.detector_factory import detector_factory


CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


class CenterNet(object):
    def __init__(self):
        rospy.init_node('centernet')
        model_path = rospy.get_param('~model_path', package_path + '/resources/models/multi_pose_dla_3x.pth')
        task = rospy.get_param('~task', 'multi_pose')  # or ct_det
        self.__threshold = rospy.get_param('~threshold', 0.7)
        opt = opts().init('{} --load_model {}'.format(task, model_path).split(' '))
        self.__detector = detector_factory[opt.task](opt)

        self.__bridge = CvBridge()
        rospy.Subscriber('~image', Image, self.callback, queue_size=10)
        self.__pub = rospy.Publisher('~output', Image, queue_size=10)

    def callback(self, msg):
        try:
            image = self.__bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logwarn(e)
            return
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ret = self.__detector.run(rgb_image)['results']
        for k in ret.keys():
            for person in ret[k]:
                if person[4] < self.__threshold:
                    continue
                cv2.rectangle(image, (int(person[0]), int(person[1])), (int(person[2]), int(person[3])), (36, 255, 12), 2, cv2.LINE_AA)
                cv2.putText(image, str(k) + ': ' + str(person[4]), (int(person[0]), int(person[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                for i in range(int(len(person[5:]) / 2)):
                    cv2.circle(image, (int(person[i * 2 + 5]), int(person[i * 2 + 6])), radius=2, color=CocoColors[i % len(CocoColors)], thickness=2)
        self.__pub.publish(self.__bridge.cv2_to_imgmsg(image, 'bgr8'))


if __name__ == '__main__':
    node = CenterNet()
    rospy.spin()
    rospy.loginfo('finished')

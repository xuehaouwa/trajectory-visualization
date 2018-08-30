"""

by Hao Xue @ 30/08/18

"""
import cv2
from modules.visual_object import VisualObject
import numpy as np
from tputils.dataprocessing.pixel_normalize import trajectory_matrix_norm
import os
from typing import List, Dict
from gv_tools.util.region import Region
from gv_tools.util import visual, pather, core
from gv_tools.tracking.track_frame import TrackFrame
from gv_tools.tracking.tracklet import Tracklet


class DrawTrajectory:
    OUTPUT_PATH = '/home/ubuntu/Desktop/trajectory-visualization/ouput/nygc_pred/'
    color = (277, 277, 48)
    OBS_COLOR = (0, 255, 255)
    PRED_COLOR = (0, 255, 0)
    GT_COLOR = (255, 0, 0)

    def __init__(self, original_video: str, obs_data: str, predicted: str, ground_truth: str, data: str):
        self.video_cap = cv2.VideoCapture(original_video)
        self.obs = np.load(obs_data)
        self.predicted = np.load(predicted)
        self.predicted = trajectory_matrix_norm(self.predicted, 720, 480, 2)
        self.gt = np.load(ground_truth)
        self.data = np.load(data)

    def process(self):
        frame_index = 0
        while self.video_cap.isOpened():
            cap, frame = self.video_cap.read()

            img = np.copy(frame)
            # self.draw_all(img, frame_index)
            if frame_index > 41077:
                img = self.draw_obs(img, frame_index)
                img = self.draw_gt(img, frame_index)
                self.write_output(img, frame_index)
            frame_index += 1

    def process_frame(self):
        pass

    def draw_all(self, img, frame_index):
        draw_flag = False
        for t in self.data:
            for p in t:
                if p[-1] <= frame_index:
                    draw_flag = True
            if draw_flag:
                for i in range(len(t)):
                    point = t[i]
                    if point[-1] < frame_index:
                        cv2.circle(img, (point[0], point[1]), 4, color=self.color, thickness=-1)
                        if i < len(t)-1:
                            cv2.line(img, (point[0], point[1]), (t[i+1][0], t[i+1][1]), color=self.color, thickness=2)

    def draw_obs(self, img, frame_index):
        draw_flag = False
        for t in self.obs:
            for p in t:
                if p[-1] <= frame_index:
                    draw_flag = True
            if draw_flag:
                for i in range(len(t)):
                    point = t[i]
                    if point[-1] < frame_index:
                        cv2.circle(img, (point[0], point[1]), 2, color=self.OBS_COLOR, thickness=-1)
                        if i < len(t)-1:
                            cv2.line(img, (point[0], point[1]), (t[i+1][0], t[i+1][1]), color=self.OBS_COLOR, thickness=1)

        return img

    def draw_gt(self, img, frame_index):
        draw_flag = False
        for index in range(len(self.gt)):
            for p in self.gt[index]:
                if p[-1] <= frame_index:
                    draw_flag = True
            if draw_flag:
                traj = self.predicted[index]
                for i in range(len(traj)):
                    point = traj[i]
                    if point[-1] < frame_index:
                        cv2.circle(img, (int(point[0]), int(point[1])), 2, color=self.PRED_COLOR, thickness=-1)
                        if i < len(traj)-1:
                            cv2.line(img, (int(point[0]), int(point[1])),
                                     (int(traj[i+1][0]), int(traj[i+1][1])), color=self.PRED_COLOR, thickness=1)
                for i in range(len(self.gt[index])):
                    point = self.gt[index][i]
                    if point[-1] < frame_index:
                        cv2.circle(img, (point[0], point[1]), 2, color=self.GT_COLOR, thickness=-1)
                        if i < len(self.gt[index])-1:
                            cv2.line(img, (point[0], point[1]),
                                     (self.gt[index][i+1][0], self.gt[index][i+1][1]),
                                     color=self.GT_COLOR, thickness=1)
        return img

    def write_output(self, img, frame_index):
        output_path = self.OUTPUT_PATH + str(frame_index).zfill(8) + '.jpg'
        cv2.imwrite(output_path, img)


d = DrawTrajectory(original_video='/home/ubuntu/Downloads/grandcentral.avi',
                   data='/home/ubuntu/Desktop/TPSPM/data/o9_p8.npy',
                   obs_data='/home/ubuntu/Desktop/TPSPM/results/NYGC_setting_o9_p8/obs.npy',
                   predicted='/home/ubuntu/Desktop/TPSPM/results/NYGC_setting_o9_p8/1_predicted.npy',
                   ground_truth='/home/ubuntu/Desktop/TPSPM/results/NYGC_setting_o9_p8/gt.npy')
d.process()

# a = np.load('/home/ubuntu/Desktop/TPSPM/results/NYGC_setting_o9_p8/gt.npy')
# print(a.shape)
# print(np.min(a[:, :, -1]))

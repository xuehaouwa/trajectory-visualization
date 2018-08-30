"""

by Hao Xue @ 30/08/18

"""
import cv2
import numpy as np
from tputils.dataprocessing.pixel_normalize import trajectory_matrix_norm


def draw_obs(img, obs, init=0):
    for j in range(9):
        for o in obs:
            obs_traj = o
            for i in range(len(obs_traj[0: j+1])):
                cv2.circle(img, (obs_traj[i][0], obs_traj[i][1]), 3, (255, 0, 0), thickness=-1)
                if i > 0:
                    cv2.line(img, (obs_traj[i - 1][0], obs_traj[i - 1][1]), (obs_traj[i][0], obs_traj[i][1]), (255, 0, 0),
                             2)
        save_path = '/home/ubuntu/Desktop/trajectory-visualization/ouput/nygc_pred/' + str(j+init).zfill(9) + '.jpg'
        cv2.imwrite(save_path, img)


def draw_gt(img, gt, pred, init=9):
    for j in range(8):
        for k in range(len(gt)):
            GT_traj = gt[k]
            pred_traj = pred[k]
            for i in range(len(GT_traj[0: j+1])):
                cv2.circle(img, (GT_traj[i][0], GT_traj[i][1]), 5, (0, 255, 0))
                if i > 0:
                    cv2.line(img, (GT_traj[i - 1][0], GT_traj[i - 1][1]), (GT_traj[i][0], GT_traj[i][1]), (0, 255, 0),
                             2)
            # draw red predicted trajectory
                cv2.circle(img, (int(pred_traj[i][0]), int(pred_traj[i][1])), 5, (0, 0, 255))
                if i > 0:
                    cv2.line(img, (int(pred_traj[i-1][0]), int(pred_traj[i-1][1])),
                             (int(pred_traj[i][0]), int(pred_traj[i][1])),
                             (0, 0, 255),
                             2)
        save_path = '/home/ubuntu/Desktop/trajectory-visualization/ouput/nygc_pred/' + str(j+init).zfill(9) + '.jpg'
        cv2.imwrite(save_path, img)


img = cv2.imread('/home/ubuntu/Desktop/TPSPM/data/back.png')
obs = np.load('/home/ubuntu/Desktop/TPSPM/results/NYGC_setting_o9_p8/obs.npy')
gt = np.load('/home/ubuntu/Desktop/TPSPM/results/NYGC_setting_o9_p8/gt.npy')
pred = np.load('/home/ubuntu/Desktop/TPSPM/results/NYGC_setting_o9_p8/1_predicted.npy')
pred = trajectory_matrix_norm(pred, 720, 480, 2)

for i in range(100):
    draw_obs(img, obs[i*3: i*3+3], init=i*17)
    draw_gt(img, gt[i*3: i*3+3], pred[i*3: i*3+3], init=i*17+9)

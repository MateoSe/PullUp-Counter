import os
from video_to_frame import video2frames
from pose_detection import pose_detection
import numpy as np
from reps import points2reps_center, centeroidnp

def train(filepath,IMPORT = False):
    train_body_points = []
    if not IMPORT:
        i = 0
        lst = os.listdir(filepath)
        lst.sort()
        for filename in lst:
            body_points = {}
            frames = video2frames(filepath + "/" + filename)
            points = pose_detection(frames, 0.1, colour=True)
            np.save('points'+str(i)+'.npy', points, allow_pickle=True)
            np.save('frames_'+str(i)+'.npy', frames, allow_pickle=True)
            i+=1
            print(i, "/", len(os.listdir(filepath)))
            # eliminamos puntos que no nos sirven
            for body, points in points.items():
                zeros = np.sum(~points.any(1))
                if zeros < len(points) / 4:
                    body_points[body] = points
            train_body_points.append(body_points)
    else:
        i = 0
        for filename in os.listdir(filepath):
            body_points = {}
            points = np.load('points'+str(i)+'.npy', allow_pickle=True)
            frames = np.load('frames_'+str(i)+'.npy', allow_pickle=True)
            i+=1
            # eliminamos puntos que no nos sirven
            for body, points in points.item().items():
                zeros = np.sum(~points.any(1))
                if zeros < len(points) / 4:
                    body_points[body] = points
            train_body_points.append(body_points)

    means = []
    for t in train_body_points:
        top_rep, down_rep, centroids = points2reps_center(t)
        top_rep = np.array(top_rep)
        down_rep = np.array(down_rep)
        for i in range(len(top_rep)):
            up_pos = centroids[top_rep[i]]
            down_pos = centroids[down_rep[i]]
            diff = down_pos-up_pos
            means.append(diff)

    return sum(means)/len(means)

def test(filename, mean_diff, tolerance):
    body_points = {}
    frames = video2frames(filename)
    points = pose_detection(frames, 0.1, colour=True)
    # eliminamos puntos que no nos sirven
    for body, points in points.items():
        zeros = np.sum(~points.any(1))
        if zeros < len(points) / 4:
            body_points[body] = points
    top_rep, down_rep, centroids = points2reps_center(body_points)
    top_rep = np.array(top_rep)
    down_rep = np.array(down_rep)
    r = 0
    for i in range(len(top_rep)):
        up_pos = centroids[top_rep[i]]
        down_pos = centroids[down_rep[i]]
        diff = down_pos - up_pos
        if mean_diff - mean_diff * tolerance < diff < mean_diff + mean_diff * tolerance:
            r += 1
    return r





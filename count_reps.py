import numpy as np
from print_sinusoid import print_sinusoid
from pose_detection import pose_detection
from reps import points2reps_center
from video_to_frame import video2frames
import os

def count(filepath):
    body_points = {}
    frames = video2frames(filepath)
    points = pose_detection(frames, 0.1, colour=True)
    # eliminamos puntos que no nos sirven
    for body, points in points.items():
        zeros = np.sum(~points.any(1))
        if zeros < len(points) / 4:
            body_points[body] = points
    top_rep, down_rep, centroids = points2reps_center(body_points)
    print_sinusoid(centroids)
    return len(top_rep)

def accuracy():
    reps = [12,12,12,14,5,10,8,7,10,10]
    i = 0
    list_body_points = []
    for filename in os.listdir("VIDS"):
        body_points = {}
        points = np.load('points' + str(i) + '.npy', allow_pickle=True)
        frames = np.load('frames_' + str(i) + '.npy', allow_pickle=True)
        i += 1
        # eliminamos puntos que no nos sirven
        for body, points in points.item().items():
            zeros = np.sum(~points.any(1))
            if zeros < len(points) / 4:
                body_points[body] = points
        list_body_points.append(body_points)

    match = 0
    for i in range(len(list_body_points)):
        top_rep, down_rep, centroids = points2reps_center(list_body_points[i])
        print("REAL REPS:",reps[i],"PREDICTED REPS:",len(top_rep))
        if len(top_rep) == reps[i]:
            match += 1
    print("ACCURACY:",match/len(list_body_points)*100)




    top_rep, down_rep, centroids = points2reps_center(body_points)


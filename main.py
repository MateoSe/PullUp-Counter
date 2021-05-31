##### IMPORTS  ####
from train_test import train,test
from count_reps import count,accuracy

if __name__ == '__main__':

    # mean_diff = train("VIDS", IMPORT=False)
    # reps = test("VIDS/F6_t.mp4", mean_diff, 0.5)
    # reps = count("VIDS/F6_t.mp4")
    # print(reps)
    accuracy()





    """

    IMPORT = True
    body_points = {}

    if IMPORT:
        points = np.load("points.npy", allow_pickle=True)
        frames = np.load("frames.npy", allow_pickle=True)
        # eliminamos puntos que no nos sirven
        for body, points in points.item().items():
            zeros = np.sum(~points.any(1))
            if zeros < len(points) / 4:
                body_points[body] = points
    else:
        frames = video2frames("VIDS/F1_t.mp4")
        points = pose_detection(frames, 0.1, colour=True)
        np.save('points.npy', points, allow_pickle=True)
        np.save('frames.npy', frames, allow_pickle=True)
        # eliminamos puntos que no nos sirven
        for body, points in points.items():
            zeros = np.sum(~points.any(1))
            if zeros < len(points) / 4:
                body_points[body] = points

    top_rep, down_rep, p = points2reps_center(body_points)

    print("OK")

    """
    """
    
    p = [x[1] for x in p]
    p = ndimage.gaussian_filter(p,sigma=2)

    plt.plot(p)
    plt.title("Centroid coordinates over axe y")
    plt.savefig("test.png")
    
    """

"""
    print("ok")
    for part in body_points:
        p1 = body_points[part]
        p = p1[~np.all(p1 == 0, axis=1)]
        plt.title("Body coordinates over axe y")
        plt.plot(p[:50,1],label=part)
    plt.legend(loc = "center right")
    plt.savefig("Body_desp2.png")
"""

"""
  # imprimir los puntos encima de los frames (centroides)
    fr = frames[:, :, :, 0]
    x = np.size(fr, 0)
    y = np.size(fr, 1)
    len = np.size(frames, 3)
    slow = 4
    n = len * slow
    ch = np.size(fr, 2)
    f = np.zeros((x, y, ch, n), np.uint8)
    pos = -1
    for i in range(len):
        a = frames[:, :, :, i]
        # a = 1*a.transpose(1,0,2)
        # a = np.zeros((540, 960, 3), np.uint8)
        # a = cv.circle(a, (v[i][1],v[i][0]), radius=5, color=(0, 0, 255), thickness=-1)
        # a[int(p[i][1]) - 5 + 125:int(p[i][1]) + 5 + 125, int(p[i][0]) - 5 - 200:int(p[i][0]) + 5 - 200, :] = np.array((0, 0, 255))
        a[int(p[i][0]) - 5 + 125:int(p[i][0]) + 5 + 125, 270- 5: 270+5, :] = np.array(
            (0, 0, 255))
        for k in range(slow):
            pos += 1
            print(pos, "/", n)
            f[:, :, :, pos] = a[:, :, :]
    frames2video(f)
"""
"""
    # imprimir los puntos encima de los frames (tots els punts)
    fr = frames[:, :, :, 0]
    x = np.size(fr, 0)
    y = np.size(fr, 1)
    len = np.size(frames, 3)
    slow = 4
    n = len * slow
    ch = np.size(fr, 2)
    f = np.zeros((x, y, ch, n), np.uint8)
    pos = -1
    for i in range(len):
        a = frames[:, :, :, i]
        # a = 1*a.transpose(1,0,2)
        # a = np.zeros((540, 960, 3), np.uint8)
        # a = cv.circle(a, (v[i][1],v[i][0]), radius=5, color=(0, 0, 255), thickness=-1)

        for p in body_points.values():
            a[int(p[i][1]) - 5 + 125:int(p[i][1]) + 5 + 125, int(p[i][0]) - 5 - 200:int(p[i][0]) + 5 - 200, :] = np.array((0, 0, 255))

        for k in range(slow):
            pos += 1
            print(pos, "/", n)
            f[:, :, :, pos] = a[:, :, :]
    frames2video(f)
"""









"""
    # calculo de posicion superior e inferior en las repeticiones
    up_frames, down_frames = points2reps(body_points)
    n = 20
    fr = frames[:, :, :, 0]
    x = np.size(fr,0)
    y = np.size(fr,1)
    ch = np.size(fr,2)
    f = np.empty((x,y,ch,(len(up_frames)+len(down_frames))*n), np.dtype('uint8'))
    pos = 0
    for x, frame in enumerate(up_frames):
        for i in range(n):
            f[:,:,:,pos] = frames[:, :, :, frame]
            pos += 1
    for x, frame in enumerate(down_frames):
        for i in range(n):
            f[:,:,:,pos] = frames[:, :, :, frame]
            pos += 1

    frames2video(f)
"""

"""
    for frame in up_frames:
        f = frames[:, :, :, frame] * 255
        img = np.array(frames[:, :, :, frame] * 255, dtype=np.uint8)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.imshow("frame", img)
        input("Press Enter to continue...")
"""

"""
    print(r["Neck"])
    x_points = r["Neck"][:, 0]
    y_points = r["Neck"][:, 1]
    y_points = np.trim_zeros(y_points)
    y_points = y_points.tolist()
    print(y_points)
    plt.plot(y_points)
    plt.show()
"""

# frames2video(frames)

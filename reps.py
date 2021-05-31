import numpy as np
from scipy import ndimage


def points2reps_center(body_points):
    static_parts = ["RElbow", "RWrist", "LElbow", "LWrist", "Background"]
    body_points2 = body_points.copy()
    for body_part in body_points2.keys():  # creamos estructuras para detectar el punto superior e inferior de las repeticiones para partes no estaticas
        if body_part in static_parts:
            body_points.pop(body_part)

    points = centroid(body_points)
    points = [x[1] for x in points]
    points = ndimage.gaussian_filter(points, sigma=1.8)

    anterior = 0
    topRep = []
    downRep = []
    up = True
    down = False
    for x, coord in enumerate(points):
        if coord != 0:  # control de los errores de la red neuronal
            if coord < anterior and up:
                up = False
                down = True
                downRep.append(x)
            elif coord > anterior and down:
                up = True
                down = False
                topRep.append(x)
            anterior = coord
    return topRep, downRep, points


def points2reps_point2point(points):
    upParts = {}
    topRep = {}
    downRep = {}

    static_parts = ["RElbow", "RWrist", "LElbow", "LWrist", "Background"]
    for body_part in points.keys():  # creamos estructuras para detectar el punto superior e inferior de las repeticiones para partes no estaticas
        if body_part not in static_parts:
            topRep[body_part] = []
            downRep[body_part] = []

    # cuando cambia la dirección de subida a bajada o de bajada a subida guardamos el frame
    # maximos y minimos locales numpy.r_[True, a[1:] < a[:-1]] & numpy.r_[a[:-1] < a[1:], True]
    anterior = 0
    for part, p in points.items():
        anterior = 0
        up = True
        down = False
        for x, coord in np.ndenumerate(p[:, 0]):
            if coord != 0:  # control de los errores de la red neuronal
                if coord < anterior and up:
                    up = False
                    down = True
                    topRep[part].append(x[0])
                elif coord > anterior and down:
                    up = True
                    down = False
                    downRep[part].append(x[0])
                anterior = coord

    # media entre los puntos superiores y los inferiores
    top_points = [x for x in topRep.values()]
    numBodyParts = len(top_points)
    down_points = [x for x in downRep.values()]
    top_points = [item for innerlist in top_points for item in innerlist]
    down_points = [item for innerlist in down_points for item in innerlist]
    top_frames = []
    down_frames = []

    # si coinciden mas partes del cuerpo que el threshold en el frame -1 x +1 nos quedamos con la posicion del frame
    top_points_copy = top_points.copy()
    for x in top_points_copy:
        if top_points.count(x) + top_points.count(x - 1) + top_points.count(x + 1) > ((numBodyParts // 3) * 2):
            top_frames.append(x)
            top_points = [i for i in top_points if i != x]
    top_frames.sort()

    down_points_copy = down_points.copy()
    for x in down_points_copy:
        if down_points.count(x) + down_points.count(x - 1) + down_points.count(x + 1) > ((numBodyParts // 3) * 2):
            down_frames.append(x)
            down_points = [i for i in down_points if i != x]
    down_frames.sort()

    return top_frames, down_frames


def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x / length, sum_y / length


def centroid(points):
    i_points = [x for x in points.values()]
    nFrames = np.size(i_points[0], 0)
    nParts = len(i_points)
    p = np.zeros((nFrames, nParts, 2), dtype="int")
    for i in range(nFrames):
        for j in range(nParts):
            point = i_points[j][i]
            p[i, j, :] = point
    ll = []

    for i in range(nFrames):
        frame = p[i, :, :]
        frame = frame[~np.all(frame == 0, axis=1)]
        center = centeroidnp(frame)
        ll.append(center)

    print("OK")

    return ll

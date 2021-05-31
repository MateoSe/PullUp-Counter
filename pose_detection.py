import cv2 as cv
import numpy as np

def pose_detection(frames, thr, colour):
    net = cv.dnn.readNetFromTensorflow("graph_opt.pb")  # red neuronal con los pesos establecidos

    Width = 368
    Height = 368

    ## parts del cos i parelles de punts unides que serán detectades
    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

    POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

    if colour:
        frame = frames[:, :, :, 0]
        n_frames = np.size(frames, 3)
        #n_frames = 20
    else:
        frame = frames[:, :, 0]
        n_frames = np.size(frames, 2)

    llBodyParts = []
    body_points = {}
    for part in BODY_PARTS.keys():
        body_points[part] = np.zeros((n_frames, 2), np.dtype('int'))
        llBodyParts.append(part)

    frWidth = np.size(frame, 0)
    frHeight = np.size(frame, 1)

    for f in range(n_frames):
        print(f)
        if colour:
            frame = frames[:, :, :, f]
            net.setInput(
                cv.dnn.blobFromImage(frame, 1.0, (frWidth, frHeight), (127.5, 127.5, 127.5), swapRB=False, crop=False))
        else:
            frame = frames[:, :, f]
            net.setInput(cv.dnn.blobFromImage(frame, 1.0, (frWidth, frHeight), 127, swapRB=False, crop=False))

        out = net.forward()
        out = out[:, :19, :, :]  # Solo utilizamos los primeros 19 puntos detectados

        assert (len(BODY_PARTS) == out.shape[1])  # c

        points = []
        for i in range(len(BODY_PARTS)):
            # La red neuronal genera mapas de calor para cada parte del cuerpo
            heatMap = out[0, i, :, :]  # nos quedamos con el "Heat Map" del elemento

            # Buscamos los maximos del mapa de calor
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frWidth * point[0]) / out.shape[3]
            y = (frHeight * point[1]) / out.shape[2]

            # solo añadimos el punto detectado si passa el threshold
            points.append((int(x), int(y)) if conf > thr else None)

        for x, point in enumerate(points):
            p = point
            if point is not None:
                a = llBodyParts[x]
                body_points[llBodyParts[x]][f, 0] = point[0]
                body_points[llBodyParts[x]][f, 1] = point[1]

    return body_points
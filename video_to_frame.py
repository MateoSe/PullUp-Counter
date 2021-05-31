import numpy as np
import cv2 as cv


def video2frames(filepath):
    framerate = 4 # ens quedem amb frames/framerate frames
    quality_red = 2 # dividim la qualitat entre quality_red
    colour = True

    video = cv.VideoCapture(filepath)
    ret, frame = video.read()  # first frame
    n_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT)) // framerate
    x_size = frame.shape[0]
    y_size = frame.shape[1]
    if colour:
        ch_size = frame.shape[2]
        frames = np.empty((x_size // quality_red, y_size // quality_red, ch_size, n_frames), np.dtype('uint8'))
    else:
        frames = np.empty((x_size // quality_red, y_size // quality_red, n_frames), np.dtype('uint8'))

    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.resizeWindow('frame', 800, 800)
    i = framerate - 1
    while (video.isOpened()):
        try:
            i += 1
            ret, frame = video.read()
            # print(frame.shape)
            if i % framerate == 0:
                frame = frame[::quality_red, ::quality_red]
                pos = int((i / framerate) - 1)
                if colour:
                    frames[:, :, :, pos] = frame
                else:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    frames[:, :, pos] = frame

                cv.imshow('frame', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        except:  # controlar quan el video s'acaba
            break

    # When everything done, release the capture
    video.release()
    cv.destroyAllWindows()
    # frame2 = np.zeros((480, 640, 3, 142), np.dtype('uint8'))
    return frames


def frames2video(frames):
    colour = True
    if colour:
        size = np.size(frames, 0), np.size(frames, 1), 3
        n_frames = np.size(frames, 3)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter('/home/mat/Escritorio/2.2/NEW/PSIV/TEST1.avi', fourcc, 20,
                             (size[1], size[0]), colour)
        for x in range(n_frames):
            data = frames[:, :, :, x]
            out.write(data)
    else:
        size = np.size(frames, 0), np.size(frames, 1), 3
        n_frames = np.size(frames, 2)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter('/home/mat/Escritorio/2.2/NEW/PSIV/FRONTAL_1_NOCOLOUR_8f_12qr.avi', fourcc, 20,
                             (size[1], size[0]), colour)
        # ch3_frames = frames = np.empty((x_size // quality_red, y_size // quality_red, ch_size, n_frames), np.dtype('uint8'))
        for x in range(n_frames):
            data = frames[:, :, x]
            out.write(data)

    out.release()


"""
frames = video2frames("FRONTAL_1.mp4")
frames2video(frames)
print("A")
"""

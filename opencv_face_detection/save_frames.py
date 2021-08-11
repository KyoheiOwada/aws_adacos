import cv2
import os

def save_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    idx = 0
    while cap.isOpened():
        idx +=1
        ret, frame = cap.read()
        if ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:
              cv2.imwrite('{}_{}.{}'.format(base_path, '0000', ext), frame)
            elif idx < cap.get(cv2.CAP_PROP_FPS):
              continue
            else:
              second = int(cap.get(cv2.CAP_PROP_POS_FRAMES)/idx)
              filled_second = str(second).zfill(4)
              cv2.imwrite('{}_{}.{}'.format(base_path, filled_second, ext), frame)
              idx = 0

        else:
            break

save_frames('./asadora.mp4', './frame', 'frame')

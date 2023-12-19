from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2 as cv
from main import read_video

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def test():
    model = YOLO('yolov8x.pt')
    video = read_video('video/footage_01.mp4')

    frame_to_cap = 0

    assert video.isOpened()

    counter = object_counter.ObjectCounter()
    region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

    counter.set_args(
        view_img=True,
        classes_names=model.names,
        reg_pts=region_points,
        draw_tracks=True
    )    

    class_to_detect = [1,2,3,5,7]

    while True:
        ret, frame = video.read()
        video.set(cv.CAP_PROP_POS_FRAMES, frame_to_cap)

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        track = model.track(
            frame, 
            classes=class_to_detect,
            persist=True,
            show=False
        )

        frame = counter.start_counting(frame, track)

        frame_to_cap += 15

    video.release()
    cv.destroyAllWindows()
        

if __name__ == "__main__":
    test()
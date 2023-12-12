from ultralytics import YOLO
import cv2 as cv
from method import remove_unnecessary_info, draw_centered_point, tensor_list_to_int, tensor_lol_to_float

def read_video(video_path:str) -> cv.VideoCapture:
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    return cap

def read_image(image_path:str):
    img = cv.imread(image_path)
    if img is None:
        print("Could not read image")
        exit()
    return img

def extract_frame(video:cv.VideoCapture):
    counter = 1
    while True:
        ret, frame = video.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv.imshow('frame', frame)

        filename = 'image/frame_' + str(counter) + '.jpg'

    
    video.release()
    cv.destroyAllWindows()

def detect_video(video):
    frame_to_cap = 0
    
    yolo = YOLO('yolov8m.pt')
    while video.isOpened():
        # Read frame
        ret, frame = video.read()
        video.set(cv.CAP_PROP_POS_FRAMES, frame_to_cap)
        
        # Out of frame, exit here
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv.resize(frame, (640, 50))

        # Predict base on frame
        results = yolo.predict(frame, device='0')

        zipped = [z for z in zip(tensor_list_to_int(results[0].boxes.cls), 
                                    tensor_list_to_int(results[0].boxes.conf),
                                    tensor_lol_to_float(results[0].boxes.xywh))]
            
        vehicle_detected = remove_unnecessary_info(zipped)

        frame = draw_centered_point(frame, vehicle_detected)

        cv.imshow('frame', frame)
        frame_to_cap += 15
        if cv.waitKey(1) == ord('q'):
            break
    
    video.release()
    cv.destroyAllWindows()

def detect_image(image):
    yolo = YOLO('yolov8s.pt')
    # Inference
    results = yolo.predict(image, show=True)

def main():
    video = cv.VideoCapture('video/Footage_01.mp4')

    detect_video(video)

if __name__ == '__main__':
    main()
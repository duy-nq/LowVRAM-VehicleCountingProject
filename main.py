from ultralytics import YOLO
import cv2 as cv
from method import draw_centered_point, tensor_list_to_int, tensor_lol_to_float, write_to_file
from draw import display_time, display_nov
import timer

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
    # Frame to capture
    frame_to_cap = 0
    frame_skip = 15
    
    # Number of vehicles in the counting space
    new_nov = 0
    total_nov = 0

    # Base on intensity of traffic, scale to determine the counting space
    scale = 0

    # Handle gap between 2 groups of vehicles
    no_nov = 0
    is_gap = True
    
    # Model to detect vehicles
    yolo = YOLO('yolov8x.pt')

    # Data to write to file
    group = 1
    tmp_time = timer.START_TIME
    data = {}
    
    while video.isOpened():
        ret, frame = video.read()
        video.set(cv.CAP_PROP_POS_FRAMES, frame_to_cap)
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Resize frame
        frame = cv.resize(frame, (1024, 576))

        # Display fake time
        frame = display_time(frame, frame_skip)

        d_space = frame[200:450, 300:650]

        # Make prediction
        results = yolo.predict(d_space, device='0', classes=[1,2,3,5,7], max_det=18, conf=0.25)
        

        zipped = [z for z in zip(tensor_list_to_int(results[0].boxes.cls), 
                                    tensor_list_to_int(results[0].boxes.conf),
                                    tensor_lol_to_float(results[0].boxes.xywh))]
        
        try:
            if (len(zipped) < 8):
                scale = 0
            elif (len(zipped) < 15):
                scale = 10
            else:
                scale = 20

            for obj in zipped:               
                if obj[2][1] >= 60 and obj[2][1] < 160:
                    is_gap = False
                if obj[2][1] >= 160+scale and obj[2][1] <= 250-scale:
                    new_nov += 1

            if (frame_to_cap == 0):
                total_nov = len(zipped)-new_nov
            else:
                total_nov += new_nov
            
            display_nov(frame, new_nov, 150, 'Count-space')
            display_nov(frame, len(zipped), 100, 'ALLNOV')
            display_nov(frame, total_nov, 200, 'Total')
        except:
            display_nov(frame, 0, 100, 'ALLNOV')
            is_gap = True

        if (is_gap == True):
            no_nov += 1

            if (no_nov == 3 and total_nov != 0):
                data['group'] = group
                data['duration'] = timer.get_time_now() - tmp_time
                data['nov'] = total_nov

                write_to_file('data.json', data)
                
                total_nov = 0
                no_nov = 0
                group += 1
                tmp_time = timer.get_time_now()
        else:
            is_gap = True
            no_nov = 0         

        new_nov = 0

        d_space = draw_centered_point(d_space, zipped)

        cv.imshow('d_space', d_space)
        cv.imshow('frame', frame)
        
        frame_to_cap += frame_skip
        if cv.waitKey(1) == ord('q'):
            break
    
    video.release()
    cv.destroyAllWindows()

def detect_image(image):
    yolo = YOLO('yolov8s.pt')
    # Inference
    results = yolo.predict(image, show=True)
    print(results)

def main():
    video = cv.VideoCapture('static/videos/Footage_04.mp4')

    detect_video(video)

if __name__ == '__main__':
    main()
import cv2 as cv
from cv2 import line, circle
import timer

def draw_a_line(image, p1: tuple, p2: tuple, color_BGR: tuple):
    return line(image, p1, p2, color_BGR, 2)

def draw_a_circle(image, point: tuple, color_BGR: tuple):
    return circle(image, point, 5, color_BGR, -1)

def crop_image(image, p1: tuple, p2: tuple):
    return image[p1[0]:p2[0], p1[1]:p2[1]]

def display_time(frame, frame_skip: int):
    timer.counter(frame_skip)

    h, m, s = timer.get_time()

    str_time = '{h}:{m}:{s}'.format(h=h, m=m, s=s)

    return cv.putText(frame, str_time, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

def display_nov(frame, nov: int, yaxis: int, title: str):
    str_nov = '{t}: {n}'.format(t=title, n=nov)
    
    return cv.putText(frame, str_nov, (10, yaxis), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

def test():
    img = cv.imread('image/Street01.png')
    
    img = draw_a_circle(img, (180, 520), (0, 0, 255))

    img = draw_a_circle(img, (720, 520), (0, 0, 255))

    img = draw_a_line(img, (180, 520), (720, 520), (0, 0, 255))

    img = cv.resize(img, (200, 400))
    
    cv.imshow('frame', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
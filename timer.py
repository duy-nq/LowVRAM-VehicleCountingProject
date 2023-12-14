import time

START_TIME = 28000
TIME_NOW = 28000

def set_time():
    global TIME_NOW
    TIME_NOW = START_TIME

def get_time():
    return time.gmtime(TIME_NOW).tm_hour, time.gmtime(TIME_NOW).tm_min, time.gmtime(TIME_NOW).tm_sec

def get_time_now():
    return TIME_NOW

def counter(frame_skip: int):
    global TIME_NOW
    TIME_NOW = TIME_NOW + frame_skip/30
    return TIME_NOW
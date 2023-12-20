import torch
import cv2 as cv
import json

UNWANTED_CLASS = [0,4,6]
COLOR_FOR_CLASS = {
    1: (0, 0, 255),    # Red (BGR)
    3: (0, 255, 0),    # Green (BGR)
    2: (255, 0, 0),    # Blue (BGR)
    5: (0, 255, 255),  # Yellow (BGR)
    7: (255, 0, 255)   # Magenta (BGR)
}

def tensor_lol_to_float(tensor_list):
    """
    Convert a list of tensors to a list of list of int
    """
    return [[float(val) for val in sub_list] for sub_list in tensor_list.tolist()]

def tensor_list_to_int(tensor_list: torch.Tensor):   
    return [int(val) for val in tensor_list.tolist()]

def remove_unnecessary_info(zipped: list):
    """
    Remove the unwanted classification from the zipped list
    """
    handle = zipped.copy()

    for i in range(len(zipped)):
        if zipped[i][0] in UNWANTED_CLASS:
            handle.remove(zipped[i])

    return handle

def calculate_centered_point(xywh: list):
    return (xywh[0], xywh[1])

def draw_centered_point(image, vd: list):
    point = [calculate_centered_point(xywh) for xywh in [xywh[2] for xywh in vd]]
    vclass = [vclass[0] for vclass in vd]

    for i in range(len(point)):
        cv.circle(image, (int(point[i][0]), int(point[i][1])), 5, COLOR_FOR_CLASS.get(vclass[i]), -1)
    
    return image

def write_to_file(file_name, data: dict):   
    with open(file_name, 'a') as f:
        f.write(json.dumps(data)+',')
    
    f.close()
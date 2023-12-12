# Attributes of return value from YOLO predict
## orig_img (numpy.ndarray): 
    The original image as a numpy array.
## orig_shape (tuple): 
    The original image shape in (height, width) format.
## boxes (Boxes, optional): 
    A Boxes object containing the detection bounding boxes.
## masks (Masks, optional): 
    A Masks object containing the detection masks.
## probs (Probs, optional): 
    A Probs object containing probabilities of each class for classification task.
## keypoints (Keypoints, optional): 
    A Keypoints object containing detected keypoints for each object.
## speed (dict): 
    A dictionary of preprocess, inference, and postprocess speeds in milliseconds per image.
## names (dict): 
    A dictionary of class names.
## path (str): 
    The path to the image file.
## _keys (tuple): 
    A tuple of attribute names for non-empty attributes.
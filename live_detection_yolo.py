import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
from utils import create_label_dict
import time

from ultralytics import YOLO



'''Detection model'''
# Path to tflite model
model_path = 'models_yolov8/best.pt'
# Model input size
input_size = (640, 640)

# Detection score threshold
det_score_thres = 0.5

# Path to id to label file
labelmap_path = 'labelmap.txt'

# Camera type and orientation
res = (640, 480)
is_picamera = True
flip = True

# Constants for box drawing
_MARGIN = 15  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 2

# Define colors for bounding boxes
colors = {0: (0, 255, 0),
          1: (255, 50, 50),
          2: (0, 0, 255),
          3: (0, 255, 255)}



def visualize_yolo(img, boxes, score_thres, label_dict, colors):

    for idx, box in enumerate(boxes):

        # Get the score
        score = box.conf[0].numpy()
        
        if score >= score_thres:

            # Get rect
            x1, y1, x2, y2 = box.xyxy[0].numpy()

            # Get class name
            try:
                class_id = int(box.cls[0])
                class_name = (label_dict[class_id]).strip()
            except:
                class_id = -1
                class_name = 'Unknown'

            # Get color
            if class_id != -1:
                color = colors[class_id]
            else:
                # Unknown class, show in white color
                color = (255,255,255)

            # Draw bounding_box
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            result_text = f'{class_name}: {str(score)[:4]}'
            text_location = (_MARGIN + int(y1), _MARGIN + _ROW_SIZE + int(x1))
            cv.putText(img, result_text, text_location, cv.FONT_HERSHEY_PLAIN, _FONT_SIZE, color, _FONT_THICKNESS)
            
    return img


def start_detection_yolo(cam, 
                         model_path,
                         input_size,
                         score_thres,
                         labelmap_path, 
                         is_picamera=True, 
                         flip=False):
    
    # Load the YOLO model
    model = YOLO(model_path)

    # Create dictionary to map class ID to class name
    id2name_dict = create_label_dict(labelmap_path)

    # Picamera
    cam.start()
    print ('Camera is running')

    
    while(True):

        try:
            t1 = time.time()

            '''Capture'''
            if is_picamera:
                # picamera
                frame_ori = cam.capture_array()
            else:
                # USB camera
                ret, frame_ori = cam.read()

            # Flip
            if flip:
                frame_ori = cv.rotate(frame_ori, cv.ROTATE_180)

            frame_ori = frame_ori[:,:,::-1]
            
            frame = frame_ori.copy()

            ''' Preprocess '''
            
            
            # Resize the frame to match the model input size
            #frame = cv.resize(frame, input_size)

            #frame = frame[:,:,::-1]

            print (frame.shape, frame.dtype)

            # ''' Run object detection '''
            results = model(frame, conf=score_thres)

            # Bounding boxes coordinates
            boxes = results[0].boxes

            if len(boxes) > 0:

                # Draw the detection result
                frame_ori = visualize_yolo(frame_ori, 
                                           boxes, 
                                           score_thres,
                                           id2name_dict,
                                           colors)
        
            # RGB to BGR for displaying
            # frame_ori = frame_ori[:,:,::-1]

            # Display the resulting frame
            cv.imshow('frame', frame_ori)

            # the 'q' button is set as the
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            t2 = time.time()
            print (f'frame_time: {t2-t1}')

        except Exception as e:
            print (e)
            # On error, release the camera object
            cam.stop()
            break

    # After the loop release the cap object
    if not is_picamera:
        cam.release()
    # Destroy all the windows
    cv.destroyAllWindows()


if __name__ == '__main__':
    
    '''Setting up and configure the camera'''
    # Picamera
    cam = Picamera2()
    config = cam.create_preview_configuration(main={"size": res, "format": "BGR888"})
    cam.configure(config)

    # USB camera
    # # Define a video capture object
    # cam = cv.VideoCapture(0)
    # cam.set(cv.CAP_PROP_FRAME_WIDTH, res[0])
    # cam.set(cv.CAP_PROP_FRAME_HEIGHT, res[1])

    # Start camera
    start_detection_yolo(cam, 
                    model_path, 
                    input_size, 
                    det_score_thres, 
                    labelmap_path,
                    is_picamera,
                    flip)

    print ('end')
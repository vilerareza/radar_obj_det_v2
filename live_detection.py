import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
from utils import visualize, create_label_dict
import time


def start_detection_efficientdet(cam, 
                    model_path,
                    input_size,
                    score_thres,
                    labelmap_path, 
                    is_picamera=True, 
                    flip=False):
    
    # Creating detector
    detector = create_detector(model_path)  
    detector_output = detector.get_output_details()
    detector_input = detector.get_input_details()[0]

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

            frame = frame_ori.copy()

            ''' Preprocess '''
            # Convert BGR to RGB
            # frame = frame[:,:,::-1]
            # Resize the frame to match the model input size
            frame = cv.resize(frame, input_size).astype('uint8')
            frame = np.expand_dims(frame, axis=0)

            # ''' Run object detection '''
            detector.set_tensor(detector_input['index'], frame)
            detector.invoke()
            # Bounding boxes coordinates
            bboxes = detector.get_tensor(detector_output[1]['index'])[0]
            # Detected objects class ID
            class_ids = detector.get_tensor(detector_output[3]['index'])[0]
            # Detection scores
            scores = detector.get_tensor(detector_output[0]['index'])[0]
            scores = [round(score, 2) for score in scores]

            # # Check the confidence based on score threshold
            # confident_detection_idx = []
            # for idx, score in enumerate(scores):
            #     if score >= score_thres:
            #         confident_detection_idx.append(idx)
    
            # print (confident_detection_idx)

            if len(bboxes) > 0:

                # Draw the detection result
                frame_ori = visualize(frame_ori, 
                                    bboxes, 
                                    class_ids, 
                                    scores, 
                                    score_thres, 
                                    id2name_dict, 
                                    color='rgb',
                                    model_type='efficientdet')
        
            frame_ori = frame_ori[:,:,::-1]

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


def start_detection_retinanet(cam, 
                    model_path,
                    input_size,
                    score_thres,
                    labelmap_path, 
                    is_picamera=True, 
                    flip=False):
    
    # Preprocessing params
    MEAN_NORM = (0.485, 0.456, 0.406)
    STDDEV_NORM = (0.229, 0.224, 0.225)
    MEAN_RGB = tuple(255 * i for i in MEAN_NORM)
    STDDEV_RGB = tuple(255 * i for i in STDDEV_NORM)
    MEDIAN_RGB = (128.0, 128.0, 128.0)

    # Creating detector
    detector = create_detector(model_path)  
    detector_output = detector.get_output_details()
    detector_input = detector.get_input_details()[0]

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

            frame_ori = cv.resize(frame_ori, input_size)
            # Resize the frame to match the model input size
            frame = frame_ori.copy().astype('float32')

            ''' Preprocess '''
            # Convert BGR to RGB
            # frame = frame[:,:,::-1]

            offset = MEAN_RGB
            offset = np.expand_dims(offset, axis=0)
            offset = np.expand_dims(offset, axis=0)
            frame -= offset

            scale = STDDEV_RGB
            scale = np.expand_dims(scale, axis=0)
            scale = np.expand_dims(scale, axis=0)
            frame /= scale

            frame = np.expand_dims(frame, axis=0)

            # ''' Run object detection '''
            detector.set_tensor(detector_input['index'], frame)
            detector.invoke()
            # Bounding boxes coordinates
            bboxes = detector.get_tensor(detector_output[1]['index'])[0]
            # Detected objects class ID
            class_id = detector.get_tensor(detector_output[2]['index'])[0]
            class_id -= 1 
            # Detection scores
            scores = detector.get_tensor(detector_output[3]['index'])[0]

            if len(bboxes) != 0:
                for i in range(len(bboxes)):
                    score = round(scores[i], 2)
                    if score >= score_thres:
                        try:
                            print (f'{(id2name_dict[class_id[i]]).strip()}, Score: {score}')
                        except:
                            print (f'Class name does not exist for label ID {class_id[i]}') 
            
            # Draw the detection result
            frame_ori = visualize(frame_ori, bboxes, class_id, scores, score_thres, id2name_dict, color='rgb')
            frame_ori = frame_ori[:,:,::-1]

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



def create_detector(model_path):
    # Initialize the object detector
    detector = tflite.Interpreter(model_path)
    # Allocate memory for the model's input `Tensor`s
    detector.allocate_tensors()
    return detector


if __name__ == '__main__':
    
    '''Detection model'''
    # Path to tflite model
    model_path = 'models_tflite/efficientdet_640.tflite'
    # Model input size
    input_size = (640, 640)
    # Model type
    model_type = 'efficientdet'
    if model_type not in ['efficientdet', 'retinanet']:
        print (' Model type not supported')
        exit()

    '''Detection score threshold'''
    det_score_thres = 0.3
    
    '''Path to id to label file'''
    labelmap_path = 'labelmap.txt'

    '''Camera type and orientation'''
    res = (640, 480)
    is_picamera = True
    flip = True

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
    if model_type == 'efficientdet':
        start_detection_efficientdet(cam, 
                        model_path, 
                        input_size, 
                        det_score_thres, 
                        labelmap_path,
                        is_picamera,
                        flip)
        
    elif model_type == 'retinanet':
        start_detection_retinanet(cam, 
                        model_path, 
                        input_size, 
                        det_score_thres, 
                        labelmap_path,
                        is_picamera,
                        flip)

    print ('end')
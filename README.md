# Instruction for Running TFLite Models on Raspberry Pi

## 1. Install the requirements.

```sh
pip install -r ./requirements.txt
```

## 2. Verify the label map.
Verify the `labelmap.txt` contains correct object classes.


## 3. To test the object detection model on single image, run test_detect.py.

```sh
python test_detect.py
```

Adjust the following variables:

* model_path: path to the tflite model to use.
* input_size: Input size of selected tflite model:
    efficient_Det_0.tflite -> (320, 320)
    efficient_Det_1.tflite -> (384, 384)
    efficient_Det_2.tflite -> (448, 448)
    efficient_Det_4.tflite -> (640, 640)
* score_threshold: detection confidence threshold
* is_picamera: set to `True` if using picamera or `False` if using webcam
* flip: set to `True` if camera is flipped or `False` if not. Check it by observing the OpenCV window.



## 4. To test the object detection model live from camera, run live_detection.py.

```sh
python live_detection.py
```

Adjust the following variables:

* is_flip: Set this to True or False depending on your camera orientation. Try this by experiment.
* model_path: path to the tflite model to use.
* input_size: Input size of selected tflite model:
    efficient_Det_0.tflite -> (320, 320)
    efficient_Det_1.tflite -> (384, 384)
    efficient_Det_2.tflite -> (448, 448)
    efficient_Det_4.tflite -> (640, 640)
* test_image_path: path to the test image file
* test_image_bbox_path: path for the resulting detection image (with bounding box)
* det_score_threshold: detection confidence threshold


## 5. Detection outputs

The following lines of codes in `live_detection.py` and `test_detect.py` is the result of detection from each frame. Use it accordingly.

```python
# Bounding boxes coordinates
bboxes = interpreter.get_tensor(interpreter_output[1]['index'])[0]
# Detected objects class ID
class_id = interpreter.get_tensor(interpreter_output[3]['index'])[0]
# Detection scores
scores = interpreter.get_tensor(interpreter_output[0]['index'])[0]
```
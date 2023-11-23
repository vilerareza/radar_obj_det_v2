# Instruction for Running TFLite Models on Raspberry Pi

## 1. Install the requirements.

```sh
pip install -U -r ./requirements.txt
```

## 2. Verify the label map.
Verify the `labelmap.txt` contains correct object classes.


## 3. Run the object detection live from camera using TFLite model.

```sh
python live_detection_tflite.py
```

Adjust the following variables:

* flip: Set this to True or False depending on your camera orientation. Try this by experiment.
* model_path: path to the tflite model to use.
* input_size: Input size of selected tflite model:
* model_type: type 'efficientdet' or 'retinanet' according to selected tflite model
* det_score_threshold: detection confidence threshold. Based on my observation, for efficientdet: Try 0.6-0.7, for retinanet: Try 0.5-0.6.


### Detection outputs

The following lines of codes in `live_detection_tflite.py` is the result of detection from each frame. Use it accordingly.

For Efficientdet

```python
# Bounding boxes coordinates
bboxes = interpreter.get_tensor(interpreter_output[1]['index'])[0]
# Detected objects class ID
class_id = interpreter.get_tensor(interpreter_output[3]['index'])[0]
# Detection scores
scores = interpreter.get_tensor(interpreter_output[0]['index'])[0]
```

For Retinanet

```python
# Bounding boxes coordinates
bboxes = interpreter.get_tensor(interpreter_output[1]['index'])[0]
# Detected objects class ID
class_id = interpreter.get_tensor(interpreter_output[2]['index'])[0]
# Detection scores
scores = interpreter.get_tensor(interpreter_output[3]['index'])[0]
```

## 4. Run the object detection live from camera using YOLO model.

```sh
python live_detection_yolo.py
```

Adjust the following variables:

* flip: Set this to True or False depending on your camera orientation. Try this by experiment.
* model_path: path to the tflite model to use.
* input_size: Input size of selected tflite model:
* det_score_threshold: detection confidence threshold. 

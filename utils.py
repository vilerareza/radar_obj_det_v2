# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to display the pose detection results."""

import cv2

_MARGIN = 15  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 2

color_blue = (50, 50, 255)  # blue
color_red = (255, 0, 0)  # red
color_green = (0, 255, 0)  # green
color_yellow = (255, 255, 0)  # yellow

color_blue_bgr = (255, 50, 50)  # blue
color_red_bgr = (0, 0, 255)  # red
color_green_bgr = (0, 255, 0)  # green
color_yellow_bgr = (0, 255, 255)  # yellow


def visualize(img, bboxes, class_id, scores, score_thres, label_dict, color='bgr', model_type = 'efficientdet'):
  # Draws bounding boxes on the input image and return it.
   
   if model_type == 'efficientdet':
      factor_w = img.shape[1]
      factor_h = img.shape[0]
   else:
      factor_w = 1
      factor_h = 1


   for i in range(len(bboxes)):

      if scores[i] >= score_thres:

         #try:

         print (label_dict[class_id[i]], scores[i])

         if class_id[i] == 0:
            if color == 'bgr':
               annotation_color = color_green_bgr
            else:
               annotation_color = color_green
         elif class_id[i] == 1:
            if color == 'bgr':
               annotation_color = color_yellow_bgr
            else:
               annotation_color = color_yellow
         elif class_id[i] == 2:
            if color == 'bgr':
               annotation_color = color_blue_bgr
            else:
               annotation_color = color_blue
         else:
            if color == 'bgr':
               annotation_color = color_red_bgr
            else:
               annotation_color = color_red
         
         # Draw bounding_box
         start_point = (int(bboxes[i][1]*factor_w), int(bboxes[i][0]*factor_h))
         end_point = int(bboxes[i][3]*factor_w ), int(bboxes[i][2]*factor_h)
         cv2.rectangle(img, start_point, end_point, annotation_color, 2)

         # Draw label and score
         class_name = (label_dict[class_id[i]]).strip()
         result_text = f'{class_name}: {str(scores[i][:4])}'
         text_location = (_MARGIN + int(bboxes[i][1]*factor_w),
                        _MARGIN + _ROW_SIZE + int(bboxes[i][0]*factor_h))
         cv2.putText(img, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                     _FONT_SIZE, annotation_color, _FONT_THICKNESS)
            
         #except:
            
         #   print (f'Class name does not exist for label ID {class_id[i]}') 

   return img


def create_label_dict(label_file_path):
  # Create a dictionary that maps class ID to class name
  label_dict = {}
  try:
    with open(label_file_path) as f:
        i = 0
        for row in f:
            label_dict[i] = row
            i+=1
  except:
     print ('Error when reading label file')
  finally:
     return label_dict

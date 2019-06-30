# Mask-Boxes Prepration and Mask-BBox Alignment 

## Functions
1. prepare masks of persons, and bboxes of persons;
2. align the mask and bbox of every person via the IOU;
3. align bboxes and body joint coordinates (figure. 9 in the [tech report](https://arxiv.org/pdf/1904.00276.pdf)).

## How to use
1. install [detectorch](https://github.com/ignacio-rocco/detectorch) following its description;
2. replace the **/lib/utils/vis.py** with **vis.py** here;
3. **demo_FPN_video_new.py** takes a set of videos as inputs and outputs the masks and bboxes of every frame.
4. **poseArrayAlign.m** takes *pose-arrays* of openpose and *boxes* of detectorch as inputs, counts the in-box joints for each boxes, and aligns each bbox with a pose-array that falls mostly in the bbox. 

## Update of the **vis.py**
1. [def return_image_mask()](https://github.com/geekfeiw/wifiperson/blob/8a8a7e8d9829892fa2dc19f4a462eee1166b5f52/dataprocessing/vis.py#L806), return masks of persons, then align with their boxes in **demo_FPN_video_new.py**

2. [def save_image_mask()](https://github.com/geekfeiw/wifiperson/blob/8a8a7e8d9829892fa2dc19f4a462eee1166b5f52/dataprocessing/vis.py#L546)
save mask of all trained objects, 80 classes (departed approaches, not recommended)

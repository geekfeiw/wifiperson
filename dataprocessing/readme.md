# Mask-Boxes Prepration and Mask-BBox Alignment 

## Functions
1. prepare masks of persons, and bboxes of persons;
2. align the mask and bboxe of every person via the IOU;
3. the aligned boxes are used to align body joint coordinates (figure. 9 in the [tech report](https://arxiv.org/pdf/1904.00276.pdf)) 

## How to use
1. install [detectorch](https://github.com/ignacio-rocco/detectorch) following its description;
2. replace the **/lib/utils/vis.py** with **vis.py** here;
3. the **demo_FPN_video_new.py** takes one video as inputs and outputs the masks and bboxes of every frame.

## The update of the **vis.py**


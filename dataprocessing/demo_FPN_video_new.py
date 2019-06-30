
# coding: utf-8

# # Imports

# In[1]:


import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sys
sys.path.insert(0, "lib/")
from utils.preprocess_sample import preprocess_sample
from utils.collate_custom import collate_custom
from utils.utils import to_cuda_variable
from utils.json_dataset_evaluator import evaluate_boxes,evaluate_masks
from model.detector import detector
import utils.result_utils as result_utils
import utils.vis as vis_utils
import skimage.io as io
from utils.blob import prep_im_for_blob,im_list_to_blob
import utils.dummy_datasets as dummy_datasets
from utils.multilevel_rois import add_multilevel_rois_for_test
import cv2
import os

from utils.selective_search import selective_search # needed for proposal extraction in Fast RCNN
from PIL import Image

torch_ver = torch.__version__[:3]


# # Parameters

# In[2]:


# COCO minival2014 dataset path
coco_ann_file='datasets/data/coco/annotations/instances_minival2014.json'
img_dir='datasets/data/coco/val2014'

# model type
model_type='mask' # change here

# pretrained model
if model_type=='mask':
    arch='resnet101'
    # https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl
    pretrained_model_file = 'files/trained_models/mask_fpn/model_final.pkl'
    use_rpn_head = True
    use_mask_head = True
elif model_type=='faster':
    arch='resnet50'
    # https://s3-us-west-2.amazonaws.com/detectron/35857389/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml.01_37_22.KSeq0b5q/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl
    pretrained_model_file = 'files/trained_models/faster/e2e_faster_rcnn_R-50-FPN_2x.pkl'
    use_rpn_head = True
    use_mask_head = False
elif model_type=='fast':
    arch='resnet50'
    # https://s3-us-west-2.amazonaws.com/detectron/36225249/12_2017_baselines/fast_rcnn_R-50-FPN_2x.yaml.08_40_18.zoChak1f/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl
    pretrained_model_file = 'files/trained_models/fast/fast_rcnn_R-50-FPN_2x.pkl'
    use_rpn_head = False
    use_mask_head = False


# # Create detector model

# In[5]:


model = detector(arch=arch,
                 detector_pkl_file=pretrained_model_file,
                 conv_body_layers=['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4'],
                 conv_head_layers='two_layer_mlp',
                 fpn_layers=['layer1','layer2','layer3','layer4'],
                 fpn_extra_lvl=True,
                 roi_height=7,
                 roi_width=7,
                 roi_spatial_scale=[0.25,0.125,0.0625,0.03125],
                 roi_sampling_ratio=2,
                 use_rpn_head = use_rpn_head,
                 use_mask_head = use_mask_head,
                 mask_head_type = '1up4convs')
model = model.cuda()


def eval_model(sample):
    class_scores, bbox_deltas, rois, img_features = model(sample['image'],
                                                          sample['proposal_coords'],
                                                          scaling_factor=sample['scaling_factors'])
    return class_scores, bbox_deltas, rois, img_features

# # Load image

# In[4]:
import glob
video_dir = '/media/delight-wifi/My Passport/Dataset/WiFiPose-Video/' # chage dir



videos = glob.glob(video_dir+'*.avi')
video_num = len(videos)



# image_fn = 'demo/33823288584_1d21cf0a26_k.jpg'

# Load image
output_dir = '/media/delight-wifi/My Passport/Dataset/video-mask/'
for video_index in range(video_num):
    

    video_fn = videos[video_index]
    video_name = video_fn[len(video_dir):]
    print(video_name)
    video_name = video_fn[len(video_dir):-4]
    outputVideo_dir = output_dir + video_name + '_mask/'
    
    if not os.path.exists(outputVideo_dir):
        os.makedirs(outputVideo_dir)
    print(video_fn)	
    video = cv2.VideoCapture(video_fn)
    frame_index = 0
    while(video.isOpened()):
        #print('hello')
        frame_index = frame_index + 1
        ret, image = video.read()
        
        if ret:


            if len(image.shape) == 2: # convert grayscale to RGB
                image = np.repeat(np.expand_dims(image,2), 3, axis=2)
            orig_im_size = image.shape
            # Preprocess image
            im_list, im_scales = prep_im_for_blob(image)
            # Build sample
            sample = {}
            # im_list_to blob swaps channels and adds stride in case of fpn
            fpn_on=True
            sample['image'] = torch.FloatTensor(im_list_to_blob(im_list,fpn_on))
            sample['scaling_factors'] = im_scales[0]
            sample['original_im_size'] = torch.FloatTensor(orig_im_size)
          # Extract proposals
            if model_type=='fast':
              # extract proposals using selective search (xmin,ymin,xmax,ymax format)
                rects = selective_search(pil_image=Image.fromarray(image),quality='f')
                sample['proposal_coords']=torch.FloatTensor(preprocess_sample().remove_dup_prop(rects)[0])*im_scales[0]
            else:
                sample['proposal_coords']=torch.FloatTensor([-1]) # dummy value
            # Convert to cuda variable
            sample = to_cuda_variable(sample)





        # # Evaluate

        # In[8]:





        # In[9]:


            if torch_ver=="0.4":
                with torch.no_grad():
                    class_scores,bbox_deltas,rois,img_features=eval_model(sample)
            else:
                class_scores,bbox_deltas,rois,img_features=eval_model(sample)

        # postprocess output:
        # - convert coordinates back to original image size,
        # - treshold proposals based on score,
        # - do NMS.
            scores_final, boxes_final, boxes_per_class = result_utils.postprocess_output(rois,
                                                                            sample['scaling_factors'],
                                                                            sample['original_im_size'],
                                                                            class_scores,
                                                                            bbox_deltas)

            if model_type=='mask':
              # compute masks
                boxes_final_multiscale = add_multilevel_rois_for_test({'rois': boxes_final*sample['scaling_factors']},'rois')
                boxes_final_multiscale_th = []
                for k in boxes_final_multiscale.keys():
                    if len(boxes_final_multiscale[k])>0 and 'rois_fpn' in k:
                        boxes_final_multiscale_th.append(Variable(torch.cuda.FloatTensor(boxes_final_multiscale[k])))
                    elif len(boxes_final_multiscale[k])==0 and 'rois_fpn' in k:
                        boxes_final_multiscale_th.append(None)
                rois_idx_restore_th = Variable(torch.cuda.FloatTensor(boxes_final_multiscale['rois_idx_restore_int32']))
                masks=model.mask_head(img_features,boxes_final_multiscale_th,rois_idx_restore_th.long())
              # postprocess mask output:
                h_orig = int(sample['original_im_size'].squeeze()[0].data.cpu().numpy().item())
                w_orig = int(sample['original_im_size'].squeeze()[1].data.cpu().numpy().item())
                cls_segms = result_utils.segm_results(boxes_per_class, masks.cpu().data.numpy(), boxes_final, h_orig, w_orig,
                                                    M=28) # M: Mask RCNN resolution
            else:
                cls_segms = None

            # sio.savemat(outputVideo_dir + str(frame_index) + '.mat', {'boxes_final':boxes_final,'cls_segms':cls_segms,'scores_final':scores_final,'boxes_per_class':boxes_per_class})

            mask = vis_utils.return_image_mask(
                image,  # BGR -> RGB for visualization
                str(frame_index),
                outputVideo_dir,
                boxes_per_class,
                cls_segms,
                None,
           #     dataset=dummy_datasets.get_coco_dataset(),
            #    box_alpha=0.3,
             #   show_class=True,
                thresh=0.7
              #  kp_thresh=2,
               # show=True
            )
            # print(boxes_per_class.shape)
            person_bb = boxes_per_class[1]
            # print(np.shape(boxes_per_class))
            boxes = []
            for person_index in range(len(person_bb)):
                if person_bb[person_index, -1] > 0.9:
                    boxes = np.concatenate((boxes, person_bb[person_index, :]), axis=0)
            #    boxes = boxes.reshape(-1, 5)
            print(video_name, frame_index)

            masks = []
            if len(boxes) > 0:
                boxes = boxes.reshape(-1, 5)
                for person_index in range(len(boxes)):
                    temp_box = np.zeros([720, 1280], dtype=np.int8)
                    h_min = int(np.ceil(boxes[person_index, 1] + 0.01) - 1)
                    h_max = int(np.floor(boxes[person_index, 3]))
                    w_min = int(np.ceil(boxes[person_index, 0] + 0.01) - 1)
                    w_max = int(np.floor(boxes[person_index, 2]))
                    temp_box[h_min:h_max, w_min:w_max] = 1
                    # temp_box[0, np.ceil(boxes[person_index, 1] + 0.01)-1:np.floor(boxes[person_index,3]), np.ceil(boxes[person_index,0]+0.01):np.floor(boxes[person_index,2]) ]=1

                    mask_num = len(mask)
                    # b = mask[0]
                    # print(b)
                    # print(np.shape(mask))
                    iou = np.zeros(mask_num)
                    for mask_index in range(mask_num):
                        iou[mask_index] = np.sum(mask[mask_index] * temp_box)
                    idx = np.argmax(iou)

                    if person_index == 0:
                        masks = mask[idx].reshape(1, 720, 1280)
                    else:
                        masks = np.concatenate((masks, mask[idx].reshape(1, 720, 1280)), axis=0)
            # if not os.path.exists('/media/delight-wifi/My Passport/Dataset/video-mask/' + video_name + '_mask'):
            #     os.mkdir('/media/delight-wifi/My Passport/Dataset/video-mask/' + video_name + '_mask')
            sio.savemat(outputVideo_dir + video_name + '_' + str(frame_index + 1) + '.mat', {'boxes': boxes, 'masks': masks})

        else:
            video.release()

print('Done!')








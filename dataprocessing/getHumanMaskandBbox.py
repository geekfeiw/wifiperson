import glob
import scipy.io as sio
import cv2
import numpy as np

file_name = '10'

frame_dir = '/data/feiw/oct17outVideo/oct17set' + file_name + '/'
frames = glob.glob(frame_dir + '*.mat')
frame_num = len(frames)/2

cap = cv2.VideoCapture('/home/feiw/detectorch/demo/oct17video/oct17set'+ file_name + '.avi')
video_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if frame_num==video_frame_num:
    print('frame equals!')
else:
    print('frame doesnot equal!')

for frame_index in range(int(frame_num)):
    bb = sio.loadmat(frame_dir + str(frame_index+1)+'.mat')
    person_bb = bb['boxes_per_class'][0,1]
    mask = sio.loadmat(frame_dir + str(frame_index+1)+'.MASK.mat')
    mask = mask['mask']

    boxes = []
    for person_index in range(len(person_bb)):
        if person_bb[person_index,-1] > 0.9:
            boxes = np.concatenate((boxes, person_bb[person_index,:]), axis=0)
#    boxes = boxes.reshape(-1, 5)
    print('oct17set'+file_name,frame_index)

    masks = []
    if len(boxes)>0:
       boxes = boxes.reshape(-1, 5)         
       for person_index in range(len(boxes)):
            temp_box = np.zeros([720,1280], dtype=np.int8)
            h_min = int(np.ceil(boxes[person_index, 1] + 0.01)-1)
            h_max = int(np.floor(boxes[person_index, 3]))
            w_min = int(np.ceil(boxes[person_index, 0] + 0.01)-1)
            w_max = int(np.floor(boxes[person_index, 2]))
            temp_box[h_min:h_max, w_min:w_max] = 1
            # temp_box[0, np.ceil(boxes[person_index, 1] + 0.01)-1:np.floor(boxes[person_index,3]), np.ceil(boxes[person_index,0]+0.01):np.floor(boxes[person_index,2]) ]=1

            mask_num = len(mask)
            iou = np.zeros(mask_num)
            for mask_index in range(mask_num):
                iou[mask_index] = np.sum(mask[mask_index,:,:] * temp_box)
            idx = np.argmax(iou)

            if person_index==0:
                masks = mask[idx,:,:].reshape(1,720,1280)
            else:
                masks = np.concatenate((masks, mask[idx,:,:].reshape(1,720,1280)), axis=0)

    sio.savemat('/data/feiw/oct17outVideo/oct17set'+file_name+'_clean/oct17set'+ file_name+'_' + str(frame_index+1)+'.mat', {'boxes':boxes, 'masks':masks})

print('oct17set'+file_name+' saved succeed!')

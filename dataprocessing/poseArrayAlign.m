clear
% folder_list = {'E:\oct17\frame_csi_hm_mask_bb_array_80train\', 'E:\oct17\frame_csi_hm_mask_bb_array_20test\', ...
%     'E:\sep12\frame_csi_hm_mask_bb_array_80train\', 'E:\sep12\frame_csi_hm_mask_bb_array_20test\'};

folder_list = {'/media/feiw/New Volume1/wifiposedata/train80/'};

color = rand([9,3]);

% for folder_name = folder_list
%     folder_name{1}
% end

for folder_name = folder_list

    files = dir([folder_name{1}, '*.mat']);
    file_num = length(files);
    
    for file_index = 1:file_num
        [folder_name{1}, files(file_index).name]
        %load([folder_name{1}, files(file_index).name], 'array', 'boxes');
        load([folder_name{1}, files(file_index).name], 'boxes');
        index = getIndex(files(file_index).name);
        if files(file_index).name(10) == '0'
            load(['/media/feiw/New Volume1/poseArray/coco/', files(file_index).name(1:10), '/',...
                files(file_index).name(1:10), '_', index, '.mat'], 'coco_pose');
        else
            load(['/media/feiw/New Volume1/poseArray/coco/', files(file_index).name(1:9), '/',...
                files(file_index).name(1:9), '_', index, '.mat'], 'coco_pose');
        end
        
        array = coco_pose;
        boxes_num = size(boxes,1);
        openpose_array_num = size(array,1);

        if boxes_num>0&&openpose_array_num>0  %%%% if mask rcnn has boxes and openpose has joints
           
            %% if image having 4 persons start 
            if ~isempty(strfind(files(file_index).name, 'four')) %% if image having 4 persons
                if size(boxes,1)>4
                   %%%%%%% important get the largest 'four' boxes
                   box_size = (boxes(:,3)-boxes(:,1)) .* (boxes(:,4)-boxes(:,2)); % boxes size by width.*height
                   [~, idx] = sort(box_size); %%%
                   boxes = boxes(idx(1:4),:); %%% get the largest 2 boxes
                   %%%%%%%
                   boxes_num = size(boxes,1);
                   openpose_array = zeros([boxes_num,18,3]); % creat a list to save cressponding array
                                     
                   %% align bounding box which contains most
                   % seleting joints
                   for boxes_index = 1:boxes_num 
                       count = zeros([1, openpose_array_num]);
                           for openpose_array_index = 1:openpose_array_num
                               %%% counting the number of in-boundingbox
                               %%% joints
                               temp = sum(double(squeeze(array(openpose_array_index,:,1:2))>boxes(boxes_index,1:2))...
                                   + double(squeeze(array(openpose_array_index,:,1:2))<boxes(boxes_index,3:4)), 2);
                                count(openpose_array_index) = sum(double(temp==4));
                                %%%% 
                           end
                           [~, idx] = max(count); %% which boundingbox ha
                       openpose_array(boxes_index,:,:) = array(idx,:,:);    
                   end
                   
                else
                   %% align bounding box which contains most
                   % seleting joints
                   openpose_array = zeros([boxes_num,18,3]); % creat a list to save cressponding array
                   for boxes_index = 1:boxes_num 
                       count = zeros([1, openpose_array_num]);
                           for openpose_array_index = 1:openpose_array_num
                               %%% counting the number of in-boundingbox
                               %%% joints
                               temp = sum(double(squeeze(array(openpose_array_index,:,1:2))>boxes(boxes_index,1:2))...
                                   + double(squeeze(array(openpose_array_index,:,1:2))<boxes(boxes_index,3:4)), 2);
                                count(openpose_array_index) = sum(double(temp==4));
                                %%%% 
                           end
                           [~, idx] = max(count); %% which boundingbox 
                       openpose_array(boxes_index,:,:) = array(idx,:,:);    
                   end
                    
                end
            
%         imshow(imresize(frame,[720,1280])); hold on;
%         
%         for i = 1:boxes_num
%            rectangle('Position', [boxes(i,1:2) boxes(i,3:4)-boxes(i,1:2)], 'EdgeColor', color(i,:));
%            scatter(squeeze(openpose_array(i,:,1)), squeeze(openpose_array(i,:,2)), 'MarkerEdgeColor', color(i,:) );
%             
%         end

            %% if image having 4 persons
            elseif ~isempty(strfind(files(file_index).name, 'five')) % if image having 5 persons
                if size(boxes,1)>5
                   %%%%%%% important get the largest 'four' boxes
                   box_size = (boxes(:,3)-boxes(:,1)) .* (boxes(:,4)-boxes(:,2)); % boxes size by width.*height
                   [~, idx] = sort(box_size); %%%
                   boxes = boxes(idx(1:5),:); %%% get the largest 2 boxes
                   %%%%%%%
                   boxes_num = size(boxes,1);
                   openpose_array = zeros([boxes_num,18,3]); % creat a list to save cressponding array
                                     
                   %% align bounding box which contains most
                   % seleting joints
                   for boxes_index = 1:boxes_num 
                       count = zeros([1, openpose_array_num]);
                           for openpose_array_index = 1:openpose_array_num
                               %%% counting the number of in-boundingbox
                               %%% joints
                               temp = sum(double(squeeze(array(openpose_array_index,:,1:2))>boxes(boxes_index,1:2))...
                                   + double(squeeze(array(openpose_array_index,:,1:2))<boxes(boxes_index,3:4)), 2);
                                count(openpose_array_index) = sum(double(temp==4));
                                %%%% 
                           end
                           [~, idx] = max(count); %% which boundingbox ha
                       openpose_array(boxes_index,:,:) = array(idx,:,:);    
                   end
                   
                else
                   %% align bounding box which contains most
                   % seleting joints
                   openpose_array = zeros([boxes_num,18,3]); % creat a list to save cressponding array
                   for boxes_index = 1:boxes_num 
                       count = zeros([1, openpose_array_num]);
                           for openpose_array_index = 1:openpose_array_num
                               %%% counting the number of in-boundingbox
                               %%% joints
                               temp = sum(double(squeeze(array(openpose_array_index,:,1:2))>boxes(boxes_index,1:2))...
                                   + double(squeeze(array(openpose_array_index,:,1:2))<boxes(boxes_index,3:4)), 2);
                                count(openpose_array_index) = sum(double(temp==4));
                                %%%% 
                           end
                           [~, idx] = max(count); %% which boundingbox 
                       openpose_array(boxes_index,:,:) = array(idx,:,:);    
                   end
                    
                end
            
        %% if image having 4 persons
            elseif ~isempty(strfind(files(file_index).name, 'two')) %% if image having 2 persons
                if size(boxes,1)>2
                   %%%%%%% important get the largest 'four' boxes
                   box_size = (boxes(:,3)-boxes(:,1)) .* (boxes(:,4)-boxes(:,2)); % boxes size by width.*height
                   [~, idx] = sort(box_size); %%%
                   boxes = boxes(idx(1:2),:); %%% get the largest 2 boxes
                   %%%%%%%
                   boxes_num = size(boxes,1);
                   openpose_array = zeros([boxes_num,18,3]); % creat a list to save cressponding array
                                     
                   %% align bounding box which contains most
                   % seleting joints
                   for boxes_index = 1:boxes_num 
                       count = zeros([1, openpose_array_num]);
                           for openpose_array_index = 1:openpose_array_num
                               %%% counting the number of in-boundingbox
                               %%% joints
                               temp = sum(double(squeeze(array(openpose_array_index,:,1:2))>boxes(boxes_index,1:2))...
                                   + double(squeeze(array(openpose_array_index,:,1:2))<boxes(boxes_index,3:4)), 2);
                                count(openpose_array_index) = sum(double(temp==4));
                                %%%% 
                           end
                           [~, idx] = max(count); %% which boundingbox ha
                       openpose_array(boxes_index,:,:) = array(idx,:,:);    
                   end
                   
                else
                   %% align bounding box which contains most
                   % seleting joints
                   openpose_array = zeros([boxes_num,18,3]); % creat a list to save cressponding array
                   for boxes_index = 1:boxes_num 
                       count = zeros([1, openpose_array_num]);
                           for openpose_array_index = 1:openpose_array_num
                               %%% counting the number of in-boundingbox
                               %%% joints
                               temp = sum(double(squeeze(array(openpose_array_index,:,1:2))>boxes(boxes_index,1:2))...
                                   + double(squeeze(array(openpose_array_index,:,1:2))<boxes(boxes_index,3:4)), 2);
                                count(openpose_array_index) = sum(double(temp==4));
                                %%%% 
                           end
                           [~, idx] = max(count); %% which boundingbox 
                       openpose_array(boxes_index,:,:) = array(idx,:,:);    
                   end
                    
                end
             
        %% if image having 3 persons
            elseif ~isempty(strfind(files(file_index).name, 'three')) %% if image having 3 persons
                if size(boxes,1)>3
                   %%%%%%% important get the largest 'four' boxes
                   box_size = (boxes(:,3)-boxes(:,1)) .* (boxes(:,4)-boxes(:,2)); % boxes size by width.*height
                   [~, idx] = sort(box_size); %%%
                   boxes = boxes(idx(1:3),:); %%% get the largest 2 boxes
                   %%%%%%%
                   boxes_num = size(boxes,1);
                   openpose_array = zeros([boxes_num,18,3]); % creat a list to save cressponding array
                                     
                   %% align bounding box which contains most
                   % seleting joints
                   for boxes_index = 1:boxes_num 
                       count = zeros([1, openpose_array_num]);
                           for openpose_array_index = 1:openpose_array_num
                               %%% counting the number of in-boundingbox
                               %%% joints
                               temp = sum(double(squeeze(array(openpose_array_index,:,1:2))>boxes(boxes_index,1:2))...
                                   + double(squeeze(array(openpose_array_index,:,1:2))<boxes(boxes_index,3:4)), 2);
                                count(openpose_array_index) = sum(double(temp==4));
                                %%%% 
                           end
                           [~, idx] = max(count); %% which boundingbox ha
                       openpose_array(boxes_index,:,:) = array(idx,:,:);    
                   end
                   
                else
                   %% align bounding box which contains most
                   % seleting joints
                   openpose_array = zeros([boxes_num,18,3]); % creat a list to save cressponding array
                   for boxes_index = 1:boxes_num 
                       count = zeros([1, openpose_array_num]);
                           for openpose_array_index = 1:openpose_array_num
                               %%% counting the number of in-boundingbox
                               %%% joints
                               temp = sum(double(squeeze(array(openpose_array_index,:,1:2))>boxes(boxes_index,1:2))...
                                   + double(squeeze(array(openpose_array_index,:,1:2))<boxes(boxes_index,3:4)), 2);
                                count(openpose_array_index) = sum(double(temp==4));
                                %%%% 
                           end
                           [~, idx] = max(count); %% which boundingbox 
                       openpose_array(boxes_index,:,:) = array(idx,:,:);    
                   end
                    
                end
            else
                   %% align bounding box which contains most
                   % seleting joints
                   openpose_array = zeros([boxes_num,18,3]); % creat a list to save cressponding array
                   for boxes_index = 1:boxes_num 
                       count = zeros([1, openpose_array_num]);
                           for openpose_array_index = 1:openpose_array_num
                               %%% counting the number of in-boundingbox
                               %%% joints
                               temp = sum(double(squeeze(array(openpose_array_index,:,1:2))>boxes(boxes_index,1:2))...
                                   + double(squeeze(array(openpose_array_index,:,1:2))<boxes(boxes_index,3:4)), 2);
                                count(openpose_array_index) = sum(double(temp==4));
                                %%%% 
                           end
                           [~, idx] = max(count); %% which boundingbox 
                       openpose_array(boxes_index,:,:) = array(idx,:,:);    
                   end
                    
            end
        end
        
%         imshow(imresize(frame,[720,1280])); hold on;
%         
%         for i = 1:boxes_num
%            rectangle('Position', [boxes(i,1:2) boxes(i,3:4)-boxes(i,1:2)], 'EdgeColor', color(i,:));
%            scatter(squeeze(openpose_array(i,:,1)), squeeze(openpose_array(i,:,2)), 'MarkerEdgeColor', color(i,:) );
%             
%         end
%         pause(0.5)
%         hold off
        

          save(['/media/feiw/New Volume1/poseArray/allignedCOCOPose/', files(file_index).name] , 'openpose_array', 'boxes');
          
%         if ~isempty(strfind(folder_name{1}, 'train'))
%             save(['wifiposedata\train80\', files(file_index).name], 'openpose_array', 'boxes');
%         else 
%             save(['wifiposedata\test20\', files(file_index).name], 'openpose_array', 'boxes');
%         end
    end  
    
    
end   

function index = getIndex(file_name)
    for i = [5,4,3]
        if ~isempty(str2num(file_name(end-3-i:end-4)))
            index = file_name(end-3-i:end-4);
            break;
        end
    end
    


end

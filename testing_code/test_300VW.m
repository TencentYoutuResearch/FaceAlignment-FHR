clear;

name_s = 'Scenario1'; % change name_s here
path_folder = ['../data/' name_s '/'];
subdir = dir(path_folder);
rmse_float_total_list = [];
rmse_rnn_total_list = [];
nrmse_float_total_list = [];
nrmse_rnn_total_list = [];

sta1_float_total_list = [];
sta2_float_total_list = [];
sta1_rnn_total_list = [];
sta2_rnn_total_list = [];

for j=1:length(subdir)
    if(isequal(subdir(j).name, '.') ||...
       isequal(subdir(j).name, '..')||...
       ~subdir(j).isdir)
        continue;
    end
    
    name = subdir(j).name;
    error_flag = 0;
    error = [];
    if exist(['./Not_Used/' name '.mat'],'file')
        load(['./Not_Used/' name '.mat']);
        error_flag = 1;
    end
    
    path = [path_folder name '/'];
    d =dir([path '/annot/*.jpg']);
    scale = 1.3; % 1.3
    
    % 300vw(train)_300vw(test)
    load(['./results/' name_s '/preds_300VWtest_float_68pts_' name '_300VW_128.mat']); 
    pts_num = 68;
    pts_float_list=x(:,1:pts_num,:);
    
    rmse_float_list = [];
    nrmse_float_list = [];
    
    pts_gt_stable = zeros(size(pts_float_list,1),pts_num*2);
    pts_hm_stable = zeros(size(pts_float_list,1),pts_num*2);
    pts_rnn_stable = zeros(size(pts_float_list,1),pts_num*2);
    counti = 1;
    for i=1:length(d)
        if any(error==i) == 0 % not in the error set
            disp(num2str(i));
            filename = d(i).name;
            im = imread([path 'annot/' d(i).name]);
            [hei_ori,wid_ori,chan] = size(im);
            FileId=fopen([path 'annot/' filename(1:end-4) '.pts']);
            npoints=textscan(FileId,'%s %f',1,'HeaderLines',1);
            points=textscan(FileId,'%f %f',npoints{2},'MultipleDelimsAsOne',2,'Headerlines',2);
            pts=cell2mat(points);
            pts(isnan(pts)) = -1;
            
            % face bounding box
            miny = min(pts(:,2));
            maxy = max(pts(:,2));
            minx = min(pts(:,1));
            maxx = max(pts(:,1));
            delta_ = max(maxx-minx,maxy-miny);
            center_x = (minx+maxx)/2;
            center_y = (miny+maxy)/2;
            
            hei = scale*(maxy-miny);
            wid = scale*(maxx-minx);
            tmp = max(hei,wid);
            hei = tmp;
            wid = tmp;
            
            length_ = hei;
            start_x  = center_x - length_/2;
            start_y  = center_y - length_/2;
            end_x = start_x + wid;
            end_y = start_y + hei;
            
            %% pad by 0
            pad_minx = ceil(min(1,start_x));
            pad_maxx = ceil(max(wid_ori,end_x));
            pad_miny = ceil(min(1,start_y));
            pad_maxy = ceil(max(hei_ori,end_y));
            
            im_ = zeros(pad_maxy-pad_miny+1,pad_maxx-pad_minx+1,3);
            if start_x<1
                sx_ = ceil(abs(start_x));
                start_x_ = 1;
            else
                sx_ = 1;
                start_x_ = start_x;
            end
            if start_y<1
                sy_ = ceil(abs(start_y));
                start_y_ = 1;
            else
                sy_ = 1;
                start_y_ = start_y;
            end
            im_(sy_:sy_+hei_ori-1,sx_:sx_+wid_ori-1,:) = (im);
            im_crop_ = im_(start_y_:start_y_+length_,start_x_:start_x_+length_,:);
            [hei_crop,wid_crop,cha_crop] = size(im_crop_);
            im_crop_ = imresize(im_crop_,[256 256]);
            
            pts_float = reshape(pts_float_list(i,:,:),[pts_num,2]);
            pts_float(:,1) = pts_float(:,1)*(wid_crop/256);
            pts_float(:,2) = pts_float(:,2)*(hei_crop/256);
            pts_float(:,1) = pts_float(:,1) + start_x;
            pts_float(:,2) = pts_float(:,2) + start_y;
            
            % normalize to rnn
            for iii=1:size(pts,1)
                pts_gt_stable(counti,iii*2-1) = pts(iii,1);
                pts_gt_stable(counti,iii*2) = pts(iii,2);
                pts_hm_stable(counti,iii*2-1) = pts_float(iii,1);
                pts_hm_stable(counti,iii*2) = pts_float(iii,2);
            end
            pts_float(isnan(pts_float)) = -1;
            counti = counti+1;
            
            [nrmse_float, rmse_float, rmse_float_v2] = compute_error( pts, pts_float, delta_);  
            rmse_float_list = [rmse_float_list rmse_float_v2]; 
            nrmse_float_list = [nrmse_float_list nrmse_float];
            
            fclose('all');
        end
        
    end
    pts_gt_stable = pts_gt_stable(1:counti-1,:);
    pts_hm_stable = pts_hm_stable(1:counti-1,:);

    nrmse_float_avg = mean(nrmse_float_list);
    disp(['nrmse_float_avg: ' num2str(nrmse_float_avg)]);
    
    nrmse_float_total_list = [nrmse_float_total_list nrmse_float_avg];
end

nrmse_float_total_avg = mean(nrmse_float_total_list);
disp(['FHR-NRMSE-TOTAL-AVG: ' num2str(nrmse_float_total_avg)]);

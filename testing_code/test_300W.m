clear;

set_name = '300W_test';
path = ['../data/' set_name '/'];
d =dir([path '/*.jpg']);
load('../data/Boundingbox.mat');
load('../data/Groundtruth.mat');
load(['./results/preds_300W_68pts_128.mat']); % FHR model
scale = 1.3; % 1.3
pts_num = 68;
counti = 1;

pts_float_list=x(:,1:pts_num,:);
rmse_float_list = [];
nrmse_float_list = [];

for i=1:length(d)
    disp(num2str(i));
    j = i+3837;
    filename = [num2str(j) '.jpg'];
    im = imread([path '/' filename]);
    [hei_ori,wid_ori,chan] = size(im);
    
    % face bounding box
    pts_ = Groundtruth(j,:);
    pts = [];
    for jj = 1:pts_num
        pts(jj,1) = pts_((jj-1)*2+1);
        pts(jj,2) = pts_((jj-1)*2+2);
    end
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

    pts_crop(:,1) = pts(:,1) - start_x;
    pts_crop(:,2) = pts(:,2) - start_y;
    pts_crop(:,1) = pts_crop(:,1)/(wid_crop/256);
    pts_crop(:,2) = pts_crop(:,2)/(hei_crop/256);
    
    pts_float = reshape(pts_float_list(i,:,:),[pts_num,2]);
    pts_float_crop = pts_float;
    pts_float(:,1) = pts_float(:,1)*(wid_crop/256);
    pts_float(:,2) = pts_float(:,2)*(hei_crop/256);
    pts_float(:,1) = pts_float(:,1) + start_x;
    pts_float(:,2) = pts_float(:,2) + start_y;

    pts_float(isnan(pts_float)) = -1;
    counti = counti+1;
    
    [nrmse_float, rmse_float, rmse_float_v2] = compute_error( pts, pts_float, delta_);
    nrmse_float_list = [nrmse_float_list nrmse_float];
    fclose('all');
end

nrmse_float_avg = mean(nrmse_float_list)*100;
disp(['FHR-NRMSE: ' num2str(nrmse_float_avg)]);

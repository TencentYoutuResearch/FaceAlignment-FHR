function [ error_per_image,error_rmse, error_rmse_v2 ] = compute_error( ground_truth_all, detected_points_all, delta_)
%compute_error
%   compute the average point-to-point Euclidean error normalized by the
%   inter-ocular distance (measured as the Euclidean distance between the
%   outer corners of the eyes)
%
%   Inputs:
%          grounth_truth_all, size: num_of_points x 2 x num_of_images
%          detected_points_all, size: num_of_points x 2 x num_of_images
%   Output:
%          error_per_image, size: num_of_images x 1


num_of_images = size(ground_truth_all,3);
num_of_points = size(ground_truth_all,1);

error_per_image = zeros(num_of_images,1);
error_rmse = zeros(num_of_images,1);
error_rmse_v2 = zeros(num_of_images,1);

for i =1:num_of_images
    detected_points      = detected_points_all(:,:,i);
    ground_truth_points  = ground_truth_all(:,:,i);
    if num_of_points == 68
        interocular_distance = norm(ground_truth_points(37,:)-ground_truth_points(46,:));
    else
        if num_of_points == 51
            interocular_distance = norm(ground_truth_points(20,:)-ground_truth_points(29,:));
        elseif num_of_points == 194
            interocular_distance = norm(ground_truth_points(146,:)-ground_truth_points(126,:));
        elseif num_of_points == 86
            interocular_distance = norm(ground_truth_points(17,:)-ground_truth_points(25,:));
        elseif num_of_points == 82
            interocular_distance = norm(ground_truth_points(17,:)-ground_truth_points(25,:));
        elseif num_of_points == 7
            interocular_distance = norm(ground_truth_points(1,:)-ground_truth_points(4,:));
        end
    end
    
    sum=0;
    for j=1:num_of_points
        sum = sum+norm(detected_points(j,:)-ground_truth_points(j,:));
    end
    error_per_image(i) = sum/(num_of_points*interocular_distance);
%     error_rmse(i)=sqrt(mean((detected_points-ground_truth_points).^2));
%     %% RMSE
    diff = detected_points(:) - ground_truth_points(:);
    sum = 0;
    for jjj=1:size(diff,1)
        sum = sum + diff(jjj,1)*diff(jjj,1);
    end
    error_rmse(i) = sqrt(sum/size(diff,1)*2)*100/delta_;
    error_rmse_v2(i) = sqrt(sum/size(diff,1));
end

end


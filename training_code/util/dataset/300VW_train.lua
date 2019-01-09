local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    local matio = require 'matio'
    self.nJoints = 68
    self.accIdxs = {1,2,3,4,5,6,11,12,15,16}
    -- 68 points
    self.flipRef = {    
        {1,17}, {2,16}, {3,15},
        {4,14}, {5,13}, {6,12},
        {7,11}, {8,10}, 
        {18,27}, {19,26}, {20,25},
        {21,24}, {22,23}, {37,46},
        {38,45}, {39,44}, {40,43},
        {41,48}, {42,47}, {32,36},
        {33,35}, {49,55}, {50,54},
        {51,53}, {61,65}, {62,64},
        {68,66}, {60,56}, {59,57},    
    }

    -- Pairs of joints for drawing skeleton
    self.skeletonRef = {{1,2,1},    {2,3,1},    {3,7,1},
                        {4,5,2},    {4,7,2},    {5,6,2},
                        {7,9,0},    {9,10,0},
                        {13,9,3},   {11,12,3},  {12,13,3},
                        {14,9,4},   {14,15,4},  {15,16,4}}

    local annot = {}
    Groundtruth = matio.load('../data/Groundtruth_300VW.mat', 'Groundtruth')
    Boundingbox = matio.load('../data/Boundingbox_300VW.mat', 'Boundingbox')



    if not opt.idxRef then
        opt.idxRef = {}
        train_num = 190384
		val_num = 1435
		total_num = 191819
		pts_num = 68
        opt.idxRef.train = torch.FloatTensor(train_num)
        annot.part=torch.FloatTensor(total_num,pts_num,2)
        annot.center=torch.IntTensor(total_num,2)
        annot.scale={}
        part=torch.DoubleTensor(pts_num,2)
        center=torch.IntTensor(2)
        --scale=torch.IntTensor(2)    
        for i=1,train_num do
            for j=1,pts_num do
                part[j][1]= Groundtruth[i][2*j-1]
                part[j][2]= Groundtruth[i][2*j] 
            end
            opt.idxRef.train[i]=i
            --print(part)
            --debug.debug()
            annot.part[i]=part
            center[1]=Boundingbox[i][1]
            center[2]=Boundingbox[i][2]
            annot.center[i]=center
            annot.scale[i]=1/Boundingbox[i][3]
        end
        local perm = torch.randperm(opt.idxRef.train:size(1)):long()
        opt.idxRef.train = opt.idxRef.train:index(1, perm)
        
        opt.idxRef.valid = torch.FloatTensor(val_num)
        for i=1,val_num do
            for j=1,pts_num do
                part[j][1]= Groundtruth[i+train_num][2*j-1]
                part[j][2]= Groundtruth[i+train_num][2*j] 
            end                
            opt.idxRef.valid[i]=i+train_num
            annot.part[i+train_num]=part
            center[1]=Boundingbox[i+train_num][1]
            center[2]=Boundingbox[i+train_num][2]
            annot.center[i+train_num]=center
            annot.scale[i+train_num]=1/Boundingbox[i+train_num][3]
        end
        -- Set up training/validation split
        torch.save(opt.save .. '/options.t7', opt)
    end
    
    self.annot = annot
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel()}

    -- For final predictions
    opt.testIters = self.nsamples.test
    opt.testBatch = 1
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)
    return paths.concat('../data/300VW_train/',ffi.string(idx..'.jpg'))
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx)
    local pts = self.annot.part[idx]:clone()
    local c = self.annot.center[idx]:clone()
    local s = self.annot.scale[idx]
    -- Small adjustment so cropping is less likely to take feet out
    --c[2] = c[2] + 15 * s
    s = s * 1.03
    return pts, c, s
end

function Dataset:normalize(idx)
    return self.annot.normalize[idx]
end

return M.Dataset


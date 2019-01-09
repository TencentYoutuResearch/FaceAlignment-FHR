-------------------------------------------------------------------------------
-- Coordinate transformation
-------------------------------------------------------------------------------
function applyFn(fn, t, t2)
    -- Apply an operation whether passed a table or tensor
    local t_ = {}
    if type(t) == "table" then
        if t2 then
            for i = 1,#t do t_[i] = applyFn(fn, t[i], t2[i]) end
        else
            for i = 1,#t do t_[i] = applyFn(fn, t[i]) end
        end
    else t_ = fn(t, t2) end
    return t_
end

function getTransform(center, scale, rot, res)
    local h = 200 * scale
    --print(scale)
    --print(h)
    --debug.debug()
    local t = torch.eye(3)

    -- Scaling
    --print(res)
    --print(h)
    t[1][1] = res / h
    t[2][2] = res / h

    -- Translation
    --print(center[1]) 
    t[1][3] = res * (-center[1] / h + .5)
    t[2][3] = res * (-center[2] / h + .5)

    -- Rotation
    if rot ~= 0 then
        rot = -rot
        local r = torch.eye(3)
        local ang = rot * math.pi / 180
        local s = math.sin(ang)
        local c = math.cos(ang)
        r[1][1] = c
        r[1][2] = -s
        r[2][1] = s
        r[2][2] = c
        -- Need to make sure rotation is around center
        local t_ = torch.eye(3)
        t_[1][3] = -res/2
        t_[2][3] = -res/2
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
    end

    return t
end

function transformPreds(coords, center, scale, res)
    local origDims = coords:size()
    coords = coords:view(-1,2)
    local newCoords = coords:clone()
    for i = 1,coords:size(1) do
        newCoords[i] = transform(coords[i], center, scale, 0, res, 1)
    end
    return newCoords:view(origDims)
end

function transform(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    --print(pt[1])
    --print(pt[2])
    --debug.debug()
    pt_[1],pt_[2] = pt[1]-1,pt[2]-1
    --print(scale)
    local t = getTransform(center, scale, rot, res)
    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_):sub(1,2)

    return new_point:int():add(1)
end

-------------------------------------------------------------------------------
-- Cropping
-------------------------------------------------------------------------------
function crop(img, center, scale, rot, res,idx)
    matio=require 'matio'
    --require 'QtLua'
    local ndim = img:nDimension()

    if ndim == 2 then img = img:view(1,img:size(1),img:size(2)) end
    local ht,wd = img:size(2), img:size(3)
    local ht_1,wd_1 = img:size(2), img:size(3)
    local tmpImg,newImg = img, torch.zeros(img:size(1), res, res)

    -- Modify crop approach depending on whether we zoom in/out
    -- This is for efficiency in extreme scaling cases
    local scaleFactor = (200 * scale) / res
    --print(scale)
    --print(scaleFactor)
    --debug.debug()
    if scaleFactor < 2 then scaleFactor = 1
    else
        local newSize = math.floor(math.max(ht,wd) / scaleFactor)
        if newSize < 2 then
           -- Zoomed out so much that the image is now a single pixel or less
           if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
           return newImg
        else
           tmpImg = image.scale(img,newSize)
           ht,wd = tmpImg:size(2),tmpImg:size(3)
        end
    end

    -- Calculate upper left and bottom right coordinates defining crop region
    local c,s = center:float()/scaleFactor, scale/scaleFactor
    local ul = transform({1,1}, c, s, 0, res, true)
    local br = transform({res+1,res+1}, c, s, 0, res, true)
    --print(ul)
    --print(br)
    --debug.debug()
    if scaleFactor >= 2 then br:add(-(br - ul - res)) end

    -- If the image is to be rotated, pad the cropped area
    local pad = math.ceil(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    if rot ~= 0 then ul:add(-pad); br:add(pad) end

    -- Define the range of pixels to take from the old image
    local old_ = {1,-1,math.max(1, ul[2]), math.min(br[2], ht+1) - 1,
                       math.max(1, ul[1]), math.min(br[1], wd+1) - 1}
    -- And where to put them in the new image
    local new_ = {1,-1,math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2],
                       math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]}

    -- Initialize new image and copy pixels over

    local newImg = torch.zeros(img:size(1), br[2] - ul[2], br[1] - ul[1])
    
    --print((old_[6] - old_[5]))
    --print((new_[6] - new_[5]))
    --print((old_[4] - old_[3]))
    --print((new_[4] - new_[3]))
    --print(old_[5])
    --print(old_[3])
    --print(new_[5])
    --print(new_[3])
    --print(#newImg)
    --print(scaleFactor)
    local x_scale=256/(math.max((old_[6] - old_[5]),(old_[4] - old_[3])))
    local y_scale=256/(math.max((old_[6] - old_[5]),(old_[4] - old_[3])))
    --image.save(idx .. '.jpg',tmpImg)
    --print(unpack(new_))
    --print(center)
    --image.display(tmpImg)
    --debug.debug()
    if not pcall(function() newImg:sub(unpack(new_)):copy(tmpImg:sub(unpack(old_))) end) then
       --error_crop=1
       print("Error occurred during crop!")
       --print(#img)
       --matio.save(idx .. '.mat',img)
       --image.display(newImg)
       --print(scaleFactor)
       --debug.debug()
    end
    --matio.save(idx .. '.mat',img)
    if rot ~= 0 then
        -- Rotate the image and remove padded area
        newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
        newImg = newImg:sub(1,-1,pad+1,newImg:size(2)-pad,pad+1,newImg:size(3)-pad):clone()
    end

    if scaleFactor < 2 then newImg = image.scale(newImg,res,res) end
    if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end

    return newImg,x_scale,y_scale,old_[5],old_[3],new_[5],new_[3]
end


function twoPointCrop(img, s, pt1, pt2, pad, res)
    local center = (pt1 + pt2) / 2
    local scale = math.max(20*s,torch.norm(pt1 - pt2)) * .007
    scale = scale * pad
    local angle = math.atan2(pt2[2]-pt1[2],pt2[1]-pt1[1]) * 180 / math.pi - 90
    return crop(img, center, scale, angle, res)
end

function compileImages(imgs, nrows, ncols, res)
    -- Assumes the input images are all square/the same resolution
    local totalImg = torch.zeros(3,nrows*res,ncols*res)
    for i = 1,#imgs do
        local r = torch.floor((i-1)/ncols) + 1
        local c = ((i - 1) % ncols) + 1
        totalImg:sub(1,3,(r-1)*res+1,r*res,(c-1)*res+1,c*res):copy(imgs[i])
    end
    return totalImg
end

-------------------------------------------------------------------------------
-- Non-maximum Suppression
-------------------------------------------------------------------------------

-- Set up max network for NMS
nms_window_size = 3
nms_pad = (nms_window_size - 1)/2
maxlayer = nn.Sequential()
if cudnn then
    maxlayer:add(cudnn.SpatialMaxPooling(nms_window_size, nms_window_size,1,1, nms_pad, nms_pad))
    maxlayer:cuda()
else
    maxlayer:add(nn.SpatialMaxPooling(nms_window_size, nms_window_size,1,1, nms_pad,nms_pad))
end
maxlayer:evaluate()

function local_maxes(hm, n, c, s, hm_idx)
    hm = torch.Tensor(1,16,64,64):copy(hm):float()
    if hm_idx then hm = hm:sub(1,-1,hm_idx,hm_idx) end
    local hm_dim = hm:size()
    local max_out
    -- First do nms
    if cudnn then
        local hmCuda = torch.CudaTensor(1, hm_dim[2], hm_dim[3], hm_dim[4])
        hmCuda:copy(hm)
        max_out = maxlayer:forward(hmCuda)
        cutorch.synchronize()
    else
        max_out = maxlayer:forward(hm)
    end

    local nms = torch.cmul(hm, torch.eq(hm, max_out:float()):float())[1]
    -- Loop through each heatmap retrieving top n locations, and their scores
    local pred_coords = torch.Tensor(hm_dim[2], n, 2)
    local pred_scores = torch.Tensor(hm_dim[2], n)
    for i = 1, hm_dim[2] do
        local nms_flat = nms[i]:view(nms[i]:nElement())
        local vals,idxs = torch.sort(nms_flat,1,true)
        for j = 1,n do
            local pt = {idxs[j] % 64, torch.ceil(idxs[j] / 64) }
            pred_coords[i][j] = transform(pt, c, s, 0, 64, true)
            pred_scores[i][j] = vals[j]
        end
    end
    return pred_coords, pred_scores
end

-------------------------------------------------------------------------------
-- Drawing functions
-------------------------------------------------------------------------------

function drawGaussian(img, pt, sigma)
    -- Draw a 2D gaussian
    -- Check that any part of the gaussian is in-bounds
    local ul = {math.floor(pt[1] - 3 * sigma), math.floor(pt[2] - 3 * sigma)}
    local br = {math.floor(pt[1] + 3 * sigma), math.floor(pt[2] + 3 * sigma)}
    -- If not, return the image as is
    if (ul[1] > img:size(2) or ul[2] > img:size(1) or br[1] < 1 or br[2] < 1) then return img end
    -- Generate gaussian
    local size = 6 * sigma + 1
    local g = image.gaussian(size) -- , 1 / size, 1)
    -- Usable gaussian range
    local g_x = {math.max(1, -ul[1]), math.min(br[1], img:size(2)) - math.max(1, ul[1]) + math.max(1, -ul[1])}
    local g_y = {math.max(1, -ul[2]), math.min(br[2], img:size(1)) - math.max(1, ul[2]) + math.max(1, -ul[2])}
    -- Image range
    local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(2))}
    local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(1))}
    assert(g_x[1] > 0 and g_y[1] > 0)
    img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):add(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    img[img:gt(1)] = 1
    return img
end

function drawLine(img,pt1,pt2,width,color)
    -- I'm sure there's a line drawing function somewhere in Torch,
    -- but since I couldn't find it here's my basic implementation
    local color = color or {1,1,1}
    local m = torch.dist(pt1,pt2)
    local dy = (pt2[2] - pt1[2])/m
    local dx = (pt2[1] - pt1[1])/m
    for j = 1,width do
        local start_pt1 = torch.Tensor({pt1[1] + (-width/2 + j-1)*dy, pt1[2] - (-width/2 + j-1)*dx})
        start_pt1:ceil()
        for i = 1,torch.ceil(m) do
            local y_idx = torch.ceil(start_pt1[2]+dy*i)
            local x_idx = torch.ceil(start_pt1[1]+dx*i)
            if y_idx - 1 > 0 and x_idx -1 > 0 and y_idx < img:size(2) and x_idx < img:size(3) then
                img:sub(1,1,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[1])
                img:sub(2,2,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[2])
                img:sub(3,3,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[3])
            end
        end 
    end
    img[img:gt(1)] = 1

    return img
end

function colorHM(x)
    -- Converts a one-channel grayscale image to a color heatmap image
    local function gauss(x,a,b,c)
        return torch.exp(-torch.pow(torch.add(x,-b),2):div(2*c*c)):mul(a)
    end
    local cl = torch.zeros(3,x:size(1),x:size(2))
    cl[1] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
    cl[2] = gauss(x,1,.5,.3)
    cl[3] = gauss(x,1,.2,.3)
    cl[cl:gt(1)] = 1
    return cl
end


-------------------------------------------------------------------------------
-- Flipping functions
-------------------------------------------------------------------------------
-- 82 points
function shuffleLR_82(x)
    local dim
    if x:nDimension() == 4 then
        dim = 2
    else
        assert(x:nDimension() == 3)
        dim = 1
    end

    local matched_parts = {
        {1,9}, {2,10}, {3,11},{4,12},{5,13},
        {6,14}, {7,15}, {8,16},
        {17,25}, {18,26},{19,27},{20,28},{21,29},
        {22,30},{23,31},{24,32},
        {81,82},
        {35,43},{36,42}, {37,41},{38,40},
        {44,50},{45,49},{46,48},{51,55},{52,54},
        {56,58},{59,61},
        {62,80},{63,79},{64,78},{65,77},{66,76},
        {67,75},{68,74},{69,73},{70,72},
    }

    for i = 1,#matched_parts do
        local idx1, idx2 = unpack(matched_parts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end

    return x
end

-- 86 points
function shuffleLR(x)
    local dim
    if x:nDimension() == 4 then
        dim = 2
    else
        assert(x:nDimension() == 3)
        dim = 1
    end

    local matched_parts = {
        {1,9}, {2,10}, {3,11},{4,12},{5,13},
        {6,14}, {7,15}, {8,16},
        {17,25}, {18,26},{19,27},{20,28},{21,29},
        {22,30},{23,31},{24,32},
        {81,82},{83,85},{84,86},
        {35,43},{36,42}, {37,41},{38,40},
        {44,50},{45,49},{46,48},{51,55},{52,54},
        {56,58},{59,61},
        {62,80},{63,79},{64,78},{65,77},{66,76},
        {67,75},{68,74},{69,73},{70,72},
    }

    for i = 1,#matched_parts do
        local idx1, idx2 = unpack(matched_parts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end

    return x
end

-- 68 points
function shuffleLR_68(x)
    local dim
    if x:nDimension() == 4 then
        dim = 2
    else
        assert(x:nDimension() == 3)
        dim = 1
    end

    local matched_parts = {
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

    for i = 1,#matched_parts do
        local idx1, idx2 = unpack(matched_parts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end

    return x
end

function flip(x)
    require 'image'
    local y = torch.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    end
    return y:typeAs(x)
end

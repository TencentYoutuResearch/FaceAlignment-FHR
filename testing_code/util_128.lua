require 'torch'
require 'xlua'
require 'nn'
require 'nnx'
require 'nngraph'
require 'image'
--require 'hdf5'
require 'sys'

require 'cunn'
require 'cutorch'
require 'cudnn'

function loadAnnotations(set)
    -- Load up a set of annotations for either: 'train', 'valid', or 'test'
    -- There is no part information in 'test'

    --local a = hdf5.open('annot/' .. set .. '.h5')
    a = {}
    annot = {}

    -- Read in annotation information from hdf5 file
    local tags = {'part','center','scale','normalize','torsoangle','visible'}
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    annot.nsamples = annot.part:size()[1]
    a:close()

    -- Load in image file names
    -- (workaround for not being able to read the strings in the hdf5 file)
    annot.images = {}
    local toIdxs = {}
    local namesFile = io.open('annot/' .. set .. '_images.txt')
    local idx = 1
    for line in namesFile:lines() do
        annot.images[idx] = line
        if not toIdxs[line] then toIdxs[line] = {} end
        table.insert(toIdxs[line], idx)
        idx = idx + 1
    end
    namesFile:close()

    -- This allows us to reference all people who are in the same image
    annot.imageToIdxs = toIdxs

    return annot
end

function getPreds(hms)
    if hms:size():size() == 3 then hms = hms:view(1, hms:size(1), hms:size(2), hms:size(3)) end

    -- Get locations of maximum activations

    --print(hm:size(1))
    --hm:view(10,16,64*64)
    local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hms:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hms:size(3)):floor():add(1)


    return preds
end

function getPreds_float_256(hm)
    sigma = 1
    if hm:size():size() == 3 then hm = hm:view(1, hm:size(1), hm:size(2), hm:size(3)) end
    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hm:size(3)):floor():add(1)
    local start_x = 0 
    local start_y = 0 
	local eps = 0.000000000000001
    for i=1,hm:size(1) do -- N
        for j=1,hm:size(2) do -- C
            x_tmp = preds[{i,j,1}]
            y_tmp = preds[{i,j,2}]
            x0 = x_tmp
            y0 = y_tmp
            -- estimate the pts
			local p0 = math.max(hm[{i,j,y_tmp,x_tmp}],eps)
			if x_tmp < hm:size(4) then
			    local p1 = math.max(hm[{i,j,y_tmp,x_tmp+1}],eps)
				x1 = x0+1
				y1 = y0
				x = math.pow(3*sigma,2) * (math.log(p1) - math.log(p0)) - (math.pow(x0,2) - math.pow(x1,2) + math.pow(y0,2) - math.pow(y1,2))/2
				if y_tmp < hm:size(3) then
				    local p2 = math.max(hm[{i,j,y_tmp+1,x_tmp}],eps)
					x2 = x0
					y2 = y0+1
					y = math.pow(3*sigma,2) * (math.log(p2) - math.log(p0)) - (math.pow(x0,2) - math.pow(x2,2) + math.pow(y0,2) - math.pow(y2,2))/2
			    else
				    local p2 = math.max(hm[{i,j,y_tmp-1,x_tmp}],eps)
					x2 = x0
                    y2 = y0-1
                    y = math.pow(3*sigma,2) * (math.log(p2) - math.log(p0)) - (math.pow(x0,2) - math.pow(x2,2) + math.pow(y0,2) - math.pow(y2,2))/2
                end
            else
                local p1 = math.max(hm[{i,j,y_tmp,x_tmp-1}],eps)
                x1 = x0-1
                y1 = y0
                x = math.pow(3*sigma,2) * (math.log(p1) - math.log(p0)) - (math.pow(x0,2) - math.pow(x1,2) + math.pow(y0,2) - math.pow(y1,2))/2
                if y_tmp < hm:size(3) then
                    local p2 = math.max(hm[{i,j,y_tmp+1,x_tmp}],eps)
                    x2 = x0
                    y2 = y0+1
                    y = math.pow(3*sigma,2) * (math.log(p2) - math.log(p0)) - (math.pow(x0,2) - math.pow(x2,2) + math.pow(y0,2) - math.pow(y2,2))/2
                else
                    local p2 = math.max(hm[{i,j,y_tmp-1,x_tmp}],eps)
                    x2 = x0
                    y2 = y0-1
                    y = math.pow(3*sigma,2) * (math.log(p2) - math.log(p0)) - (math.pow(x0,2) - math.pow(x2,2) + math.pow(y0,2) - math.pow(y2,2))/2
                end
            end
            preds[{i,j,1}] = x
            preds[{i,j,2}] = y
		end
    end
	--[[
    -- set -1 to unvisible points
    local actThresh = 0.002
	for i = 1,hm:size(1) do
        for j = 1,hm:size(2) do
		    if hm[1][j]:mean() < actThresh then
			    preds[i][j][1] = -1
			    preds[i][j][2] = -1
			end
		end
	end
	]]--
    local predMask = max:gt(0):repeatTensor(1, 1, 2):float()
    preds:add(-1):cmul(predMask):add(1)
    return preds
end

function getPreds_float(hm)
    sigma = 1
    if hm:size():size() == 3 then hm = hm:view(1, hm:size(1), hm:size(2), hm:size(3)) end
    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hm:size(3)):floor():add(1)
    local start_x = 0 
    local start_y = 0 
	local eps = 0.000000000000001
    for i=1,hm:size(1) do -- N
        for j=1,hm:size(2) do -- C
            x_tmp = preds[{i,j,1}]
            y_tmp = preds[{i,j,2}]
            x0 = x_tmp
            y0 = y_tmp
            -- estimate the pts
			local p0 = math.max(hm[{i,j,y_tmp,x_tmp}],eps)
			if x_tmp < hm:size(4) then
			    local p1 = math.max(hm[{i,j,y_tmp,x_tmp+1}],eps)
				x1 = x0+1
				y1 = y0
				x = math.pow(3*sigma,2) * (math.log(p1) - math.log(p0)) - (math.pow(x0,2) - math.pow(x1,2) + math.pow(y0,2) - math.pow(y1,2))/2
				if y_tmp < hm:size(3) then
				    local p2 = math.max(hm[{i,j,y_tmp+1,x_tmp}],eps)
					x2 = x0
					y2 = y0+1
					y = math.pow(3*sigma,2) * (math.log(p2) - math.log(p0)) - (math.pow(x0,2) - math.pow(x2,2) + math.pow(y0,2) - math.pow(y2,2))/2
			    else
				    local p2 = math.max(hm[{i,j,y_tmp-1,x_tmp}],eps)
					x2 = x0
                    y2 = y0-1
                    y = math.pow(3*sigma,2) * (math.log(p2) - math.log(p0)) - (math.pow(x0,2) - math.pow(x2,2) + math.pow(y0,2) - math.pow(y2,2))/2
                end
            else
                local p1 = math.max(hm[{i,j,y_tmp,x_tmp-1}],eps)
                x1 = x0-1
                y1 = y0
                x = math.pow(3*sigma,2) * (math.log(p1) - math.log(p0)) - (math.pow(x0,2) - math.pow(x1,2) + math.pow(y0,2) - math.pow(y1,2))/2
                if y_tmp < hm:size(3) then
                    local p2 = math.max(hm[{i,j,y_tmp+1,x_tmp}],eps)
                    x2 = x0
                    y2 = y0+1
                    y = math.pow(3*sigma,2) * (math.log(p2) - math.log(p0)) - (math.pow(x0,2) - math.pow(x2,2) + math.pow(y0,2) - math.pow(y2,2))/2
                else
                    local p2 = math.max(hm[{i,j,y_tmp-1,x_tmp}],eps)
                    x2 = x0
                    y2 = y0-1
                    y = math.pow(3*sigma,2) * (math.log(p2) - math.log(p0)) - (math.pow(x0,2) - math.pow(x2,2) + math.pow(y0,2) - math.pow(y2,2))/2
                end
            end
            preds[{i,j,1}] = x
            preds[{i,j,2}] = y
		end
    end
    -- set -1 to unvisible points
    local actThresh = 0.002
	for i = 1,hm:size(1) do
        for j = 1,hm:size(2) do
		    if hm[1][j]:mean() < actThresh then
			    preds[i][j][1] = -1
			    preds[i][j][2] = -1
			end
		end
	end
    local predMask = max:gt(0):repeatTensor(1, 1, 2):float()
    preds:add(-1):cmul(predMask):add(1)
    return preds
end

function postprocess_float_256(output)
    local tmpOutput
    if type(output) == 'table' then tmpOutput = output[#output]
    else tmpOutput = output end
    local p = getPreds_float_256(tmpOutput)

    return p,tmpOutput
end

function postprocess_float(output)
    local tmpOutput
    if type(output) == 'table' then tmpOutput = output[#output]
    else tmpOutput = output end
    local p = getPreds_float(tmpOutput)

    return p,tmpOutput
end

function getPreds_1(hm)
    if hm:size():size() == 3 then hm = hm:view(1, hm:size(1), hm:size(2), hm:size(3)) end
    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hm:size(3)):floor():add(1)
    local predMask = max:gt(0):repeatTensor(1, 1, 2):float()
    preds:add(-1):cmul(predMask):add(1)
    return preds
end

function postprocess(output)
    local tmpOutput
    if type(output) == 'table' then tmpOutput = output[#output]
    else tmpOutput = output end
    local p = getPreds_1(tmpOutput)
    local scores = torch.zeros(p:size(1),p:size(2),1):float()
    --print(#tmpOutput)
    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1,p:size(1) do
        for j = 1,p:size(2) do
            local hm = tmpOutput[i][j]
            local pX,pY = p[i][j][1], p[i][j][2]
            scores[i][j] = hm[pY][pX]
            if pX > 1 and pX < 128 and pY > 1 and pY < 128 then
               local diff = torch.FloatTensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
               p[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    
    p:add(0.5)
    return p,tmpOutput
end
-------------------------------------------------------------------------------
-- Functions for setting up the demo display
-------------------------------------------------------------------------------

function drawSkeleton(input, hms, coords)

    local im = input:clone()

    local pairRef = {
        {1,2},      {2,3},      {3,7},
        {4,5},      {4,7},      {5,6},
        {7,9},      {9,10},
        {14,9},     {11,12},    {12,13},
        {13,9},     {14,15},    {15,16}
    }

    local partNames = {'RAnk','RKne','RHip','LHip','LKne','LAnk',
                       'Pelv','Thrx','Neck','Head',
                       'RWri','RElb','RSho','LSho','LElb','LWri'}
    local partColor = {1,1,1,2,2,2,0,0,0,0,3,3,3,4,4,4}

    local actThresh = 0.002
    --print(coords)
    -- Loop through adjacent joint pairings
    for i = 1,#pairRef do
        if hms[pairRef[i][1]]:mean() > actThresh and hms[pairRef[i][2]]:mean() > actThresh then
            -- Set appropriate line color
            local color
            if partColor[pairRef[i][1]] == 1 then color = {0,.3,1}
            elseif partColor[pairRef[i][1]] == 2 then color = {1,.3,0}
            elseif partColor[pairRef[i][1]] == 3 then color = {0,0,1}
            elseif partColor[pairRef[i][1]] == 4 then color = {1,0,0}
            else color = {.7,0,.7} end

            -- Draw line
            im = drawLine(im, coords[pairRef[i][1]], coords[pairRef[i][2]], 4, color, 0)
        end
    end

    return im
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

function drawLandmarks_82(input, hms, coords)
    local im = input:clone()
    --print(#hms)

    local partColor = {1,1,1,2,2,2,0,0,0,0,3,3,3,4,4,4}
    local actThresh = 0.002
    --print(#im)
    -- Loop through adjacent joint pairings
    --print(#coords)
    for i = 1,82 do
        --print(#im)
        if hms[i]:mean() > -10 then
            -- Set appropriate line color
            local color
            color = {1,0,0} 

            -- Draw line
            --print(#im)
            --print(i)
            --print(pairRef[i])
            coords_x1=torch.Tensor(2)
            coords_x2=torch.Tensor(2)
            --print(#coords[i])
            coords_x1:copy(coords[i])
            coords_x1[1]=coords_x1[1]-2
            coords_x2:copy(coords[i])
            coords_x2[1]=coords_x2[1]+2

            coords_y1=torch.Tensor(2)
            coords_y2=torch.Tensor(2)
            coords_y1:copy(coords[i])
            coords_y1[2]=coords_y1[2]-2
            coords_y2:copy(coords[i])
            coords_y2[2]=coords_y2[2]+2
            --print(coords_x1)
            --print(coords_x2)
            --debug.debug()
            im = drawLine(im, coords_x1, coords_x2, 4, color, 0)
            im = drawLine(im, coords_y1, coords_y2, 4, color, 0)
        end
    end

    return im
end

function drawLandmarks(input, coords)

    local im = input:clone()
    --print(#hms)



    local partColor = {1,1,1,2,2,2,0,0,0,0,3,3,3,4,4,4}

    local actThresh = 0.002
    --print(#im)
    -- Loop through adjacent joint pairings
    --print(#coords)
    for i = 1,86 do
        --print(#im)
        --if hms[i]:mean() > -10 then
            -- Set appropriate line color
            local color
            color = {1,0,0} 

            -- Draw line
            --print(#im)
            --print(i)
            --print(pairRef[i])
            coords_x1=torch.Tensor(2)
            coords_x2=torch.Tensor(2)
            --print(#coords[i])
            coords_x1:copy(coords[i])
            coords_x1[1]=coords_x1[1]-2
            coords_x2:copy(coords[i])
            coords_x2[1]=coords_x2[1]+2

            coords_y1=torch.Tensor(2)
            coords_y2=torch.Tensor(2)
            coords_y1:copy(coords[i])
            coords_y1[2]=coords_y1[2]-2
            coords_y2:copy(coords[i])
            coords_y2[2]=coords_y2[2]+2
            --print(coords_x1)
            --print(coords_x2)
            --debug.debug()
            im = drawLine(im, coords_x1, coords_x2, 4, color, 0)
            im = drawLine(im, coords_y1, coords_y2, 4, color, 0)
        --end
    end

    return im
end
function drawOutput(input, hms, coords)
    local im = drawLandmarks(input, hms, coords)
    local colorHms = {}
    --print(1)
    im=im:float()
    --print(im)
    --local inp64 = image.scale(a,64):mul(.3)
    local inp64 = image.scale(im,128):mul(.3)
    for i = 1,16 do 
        colorHms[i] = colorHM(hms[i])
        --print(inp64)
        
        colorHms[i]:mul(.7):add(inp64:double())
    end
    --print(colorHms[1])
    local totalHm = compileImages({colorHms[1],colorHms[2],colorHms[3],colorHms[4]},2, 2, 128)
    --print(#im)
    --print(#totalHM)
    --im = image.scale(im,1024)
    --print(#im)
    --print(#totalHm)
    im = compileImages({im,totalHm}, 1, 2, 256)
    im = image.scale(im,756)
    return im
end

function drawOutput_ori(input, coords)
    local im = drawLandmarks(input, coords)
    local colorHms = {}
    --print(1)
    im=im:float()
    --print(im)
    --local inp64 = image.scale(a,64):mul(.3)
    --local inp64 = image.scale(im,128):mul(.3)
    --for i = 1,16 do 
    --    colorHms[i] = colorHM(hms[i])
    --    --print(inp64)
    --    
    --    colorHms[i]:mul(.7):add(inp64:double())
    --end
    ----print(colorHms[1])
    --local totalHm = compileImages({colorHms[1],colorHms[2],colorHms[3],colorHms[4]},2, 2, 128)
    ----print(#im)
    ----print(#totalHM)
    ----im = image.scale(im,1024)
    ----print(#im)
    ----print(#totalHm)
    --im = compileImages({im,totalHm}, 1, 2, 256)
    --im = image.scale(im,756)
    return im
end
-------------------------------------------------------------------------------
-- Functions for evaluation
-------------------------------------------------------------------------------

function calcDists(preds, label, normalize)
    local dists = torch.Tensor(preds:size(2), preds:size(1))
    local diff = torch.Tensor(2)
    for i = 1,preds:size(1) do
        for j = 1,preds:size(2) do
            if label[i][j][1] > 1 and label[i][j][2] > 1 then
                dists[j][i] = torch.dist(label[i][j],preds[i][j])/normalize[i]
            else
                dists[j][i] = -1
            end
        end
    end
    return dists
end

function distAccuracy(dists, thr)
    -- Return percentage below threshold while ignoring values with a -1
    if not thr then thr = .5 end
    if torch.ne(dists,-1):sum() > 0 then
        return dists:le(thr):eq(dists:ne(-1)):sum() / dists:ne(-1):sum()
    else
        return -1
    end
end

function displayPCK(dists, part_idx, label, title, show_key)
    -- Generate standard PCK plot
    if not (type(part_idx) == 'table') then
        part_idx = {part_idx}
    end

    curve_res = 11
    num_curves = #dists
    local t = torch.linspace(0,.5,curve_res)
    local pdj_scores = torch.zeros(num_curves, curve_res)
    local plot_args = {}
    print(title)
    for curve = 1,num_curves do
        for i = 1,curve_res do
            t[i] = (i-1)*.05
            local acc = 0.0
            for j = 1,#part_idx do
                acc = acc + distAccuracy(dists[curve][part_idx[j]], t[i])
            end
            pdj_scores[curve][i] = acc / #part_idx
        end
        plot_args[curve] = {label[curve],t,pdj_scores[curve],'-'}
        print(label[curve],pdj_scores[curve][curve_res])
    end

    require 'gnuplot'
    gnuplot.raw('set title "' .. title .. '"')
    if not show_key then gnuplot.raw('unset key') 
    else gnuplot.raw('set key font ",6" right bottom') end
    gnuplot.raw('set xrange [0:.5]')
    gnuplot.raw('set yrange [0:1]')
    gnuplot.plot(unpack(plot_args))
end

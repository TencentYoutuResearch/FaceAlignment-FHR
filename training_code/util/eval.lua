-------------------------------------------------------------------------------
-- Helpful functions for evaluation
-------------------------------------------------------------------------------

function loadPreds(predFile, doHm, doInp)
    local f = hdf5.open(projectDir .. '/exp/' .. predFile .. '.h5','r')
    local inp,hms
    local idxs = f:read('idxs'):all()
    local preds = f:read('preds'):all()
    if doHm then hms = f:read('heatmaps'):all() end
    if doInp then inp = f:read('input'):all() end
    return idxs, preds, hms, inp
end

function calcDists(preds, label, normalize)
    local dists = torch.Tensor(preds:size(2), preds:size(1))
    local diff = torch.Tensor(2)
    --print(preds:size(1) )
    --print(preds:size(2) )
    for i = 1,preds:size(1) do
        for j = 1,preds:size(2) do
            if label[i][j][1] > 1 and label[i][j][2] > 1 then
                --print(label[i][j])
                --print(preds[i][j])
                --print(normalize)
                dists[j][i] = torch.dist(label[i][j],preds[i][j])/normalize[i]
            else
                dists[j][i] = -1
            end
        end
    end
    return dists
end

function getPreds_im(hms, center, scale)
    --debug.debug()
    if hms:size():size() == 3 then hms = hms:view(1, hms:size(1), hms:size(2), hms:size(3)) end

    -- Get locations of maximum activations
    local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hms:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hms:size(3)):floor():add(.5)
    --print(preds)
    -- Get transformed coordinates
    --print(center[{{1},{}}]:view(2))
    --print(scale[1])
    local preds_tf = torch.zeros(preds:size())
    for i = 1,hms:size(1) do        -- Number of samples
        for j = 1,hms:size(2) do    -- Number of output heatmaps for one sample
            preds_tf[i][j] = transform(preds[i][j],center[{{i},{}}]:view(2),scale[i],0,hms:size(3),true)
            --debug.debug()
        end
        --preds_tf[i] = transformPreds(preds[i], center[{{i},{}}]:view(2),scale[i],64)
    end

    return preds, preds_tf
end

function getPreds_float(hm,sigma)
    assert(hm:size():size() == 4, 'Input must be 4-D tensor')
    --print(hm:size(1))
    --hm:view(10,16,64*64)
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
            local p0 = math.max(hm[{i,j,y_tmp,x_tmp}], eps)
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
    return preds
end

function getPreds(hm)
   -- print(hm)
    --debug.debug()
    --tmpOut=tmpOut:float()
    --debug.debug()
    assert(hm:size():size() == 4, 'Input must be 4-D tensor')
    --print(hm:size(1))
    --hm:view(10,16,64*64)
    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hm:size(3)):floor():add(1)
    return preds
end

function distAccuracy(dists, thr)
    -- Return percentage below threshold while ignoring values with a -1
    --local preds = getPreds(output)
    --local gt = getPreds(label)
    --local dists = calcDists(preds, gt, torch.ones(preds:size(1))*opt.outputRes/10)
    
    if not thr then thr = .5 end
    --print('debug')
    --debug.debug()
    --print(torch.ne(dists,-1):sum())
    --print(dists)
    if torch.ne(dists,-1):sum() > 0 then
        --print(dists:le(thr))
        --print(dists:le(thr):eq(dists:ne(-1)):sum())
        --print(dists:ne(-1):sum())
        --debug.debug()
        return dists:le(thr):eq(dists:ne(-1)):sum() / dists:ne(-1):sum()
    else
        return -1
    end
end

function heatmapAccuracy(output, label, thr, idxs)
    -- Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
    -- First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    --print(idxs[1])
    local preds = getPreds(output)
    local gt = getPreds(label)
    local dists = calcDists(preds, gt, torch.ones(preds:size(1))*opt.outputRes/10)
    local acc = {}
    local avgAcc = 0.0
    local badIdxCount = 0

    if not idxs then
        --print('not')
        for i = 1,dists:size(1) do
            acc[i+1] = distAccuracy(dists[i])
    	    if acc[i+1] >= 0 then avgAcc = avgAcc + acc[i+1]
            else badIdxCount = badIdxCount + 1 end
        end
        acc[1] = avgAcc / (dists:size(1) - badIdxCount)
    else
        --print('yes')
        --print(#idxs)
        for i = 1,#idxs do
            acc[i+1] = distAccuracy(dists[idxs[i]])
	    if acc[i+1] >= 0 then avgAcc = avgAcc + acc[i+1]
            else badIdxCount = badIdxCount + 1 end
        end
        acc[1] = avgAcc / (#idxs - badIdxCount)
    end
    return unpack(acc)
end

function basicAccuracy(output, label, thr)
    -- Calculate basic accuracy
    if not thr then thr = .5 end -- Default threshold of .5
    output = output:view(output:numel())
    label = label:view(label:numel())

    local rounded_output = torch.ceil(output - thr):typeAs(label)
    local eql = torch.eq(label,rounded_output):typeAs(label)

    return eql:sum()/output:numel()
end

function displayPCK(dists, part_idx, label, title, show_key)
    -- Generate standard PCK plot

    if not (type(part_idx) == 'table') then
        part_idx = {part_idx}
    end

    curve_res = 11
    num_curves = #dists
    --print(num_curves)
    local t = torch.linspace(0,.5,curve_res)
    local pdj_scores = torch.zeros(num_curves, curve_res)
    local plot_args = {}
    print(title)
    for curve = 1,num_curves do
        for i = 1,curve_res do
            t[i] = (i-1)*.05
            local acc = 0.0
            for j = 1,#part_idx do
                --debug.debug()
                print(dists[curve][part_idx[j]])
                --debug.debug()
                acc = acc + distAccuracy(dists[curve][part_idx[j]], t[i])
            end
            pdj_scores[curve][i] = acc / #part_idx
        end
        plot_args[curve] = {label[curve],t,pdj_scores[curve],'-'}
        print(label[curve],pdj_scores[curve][curve_res])
    end
    --print(pdj_scores)
    --debug.debug()
    require 'gnuplot'
    gnuplot.raw('set title "' .. title .. '"')
    if not show_key then gnuplot.raw('unset key') 
    else gnuplot.raw('set key font ",6" right bottom') end
    gnuplot.raw('set xrange [0:.5]')
    gnuplot.raw('set yrange [0:1]')
    gnuplot.plot(unpack(plot_args))
end

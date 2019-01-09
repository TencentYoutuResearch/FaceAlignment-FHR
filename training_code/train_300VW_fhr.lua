-- Prepare tensors for saving network output
local validSamples = opt.validIters * opt.validBatch
saved = {idxs = torch.Tensor(validSamples),
         preds = torch.Tensor(validSamples, unpack(ref.predDim))}
if opt.saveInput then saved.input = torch.Tensor(validSamples, unpack(ref.inputDim)) end
if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(validSamples, unpack(ref.outputDim[1])) end
--local dataset = paths.dofile('util/dataset/' .. opt.dataset .. '.lua')
-- Main processing step
distss={}
labels={}
function step(tag)

    local avgLoss1, avgLoss2, avgLoss3= 0.0, 0.0 , 0.0
    local avgAcc=0.0
    local acc=0.0
    local output, err, idx
    local param, gradparam = model:getParameters()
    local function evalFn(x) return criterion.output, gradparam end
    
    if tag == 'train' then
        model:training()
        set = 'train'
    else
        model:evaluate()
        if tag == 'predict' then
            print("==> Generating predictions...")
            local nSamples = dataset:size('test')
            --ref.predDim = {dataset.nJoints,5}
            saved = {idxs = torch.Tensor(nSamples),
                     preds_hm = torch.Tensor(nSamples, unpack(ref.predDim)),
                     preds_img = torch.Tensor(nSamples, unpack(ref.predDim)),
                     }
            if opt.saveInput then saved.input = torch.Tensor(nSamples, unpack(ref.inputDim)) end
            if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(nSamples, unpack(ref.outputDim[1])) end
            set = 'test'
        else
            set = 'valid'
        end 
    end

    local nIters = opt[set .. 'Iters']

    error_crop=0
    RMSE=0
    for i,sample in loader[set]:run() do
        xlua.progress(i, nIters)
        local input, label_1, indices = unpack(sample)
        if tag == 'train' then
            set='train'
            batch=opt.trainBatch
        else
            batch=opt.validBatch
            set='valid'
        end
        
        if opt.GPU ~= -1 then
            -- Convert to CUDA
            input = applyFn(function (x) return x:cuda() end, input)
            label_1 = applyFn(function (x) return x:cuda() end, label_1)
        end
        local label_hm=label_1[opt.nStack]
  
        label_hm = applyFn(function (x) return x:cuda() end, label_hm)    
        --torch.cat(label_1[2], occlusion_label, 2)
        label = {}
        for i=1,opt.nStack do
            table.insert(label, label_hm)
        end

        -- Do a forward pass and calculate loss
        output = model:forward(input)
        if tag == 'train' then
            model:zeroGradParameters()
		    model:backward(input, criterion:backward(output, label))
            --grad_out=criterion:backward(output, label)
            optfn(evalFn, param, optimState)
            --print(#label_1)
            local hm = label[opt.nStack]:float()
            hm[hm:lt(0)] = 0
        else
	    -- for validation

        end

        output_2=output[opt.nStack]
        label_2=label[opt.nStack]
        local pts_num = 68
        output_1=torch.CudaTensor(batch,pts_num,opt.outputRes,opt.outputRes)
        label_1=torch.CudaTensor(batch,pts_num,opt.outputRes,opt.outputRes)
        for kkk=1,pts_num do
            output_1[{{},{kkk},{},{}}]:copy(output_2[{{},{kkk},{},{}}])
            label_1[{{},{kkk},{},{}}]:copy(label_2[{{},{kkk},{},{}}])
        end

	    if tag == 'train' then
            --[[
            preds = getPreds_float(output_1,opt.hmGauss)
            gt = getPreds_float(label_1,opt.hmGauss)
            --acc=heatmapAccuracy(output_1, label_1, 1)
	        for k=1,batch do
                local sum=0;
                local interocular_distance = torch.norm(gt[{{k},{17},{}}]-gt[{{k},{25},{}}])
                for j=1,pts_num do
                   sum = sum+torch.norm(gt[{{k},{j},{}}]-preds[{{k},{j},{}}])
                end
                acc = sum/(pts_num*interocular_distance)/batch;
            end
            ]]--

        else
            preds = getPreds_float(output_1,opt.hmGauss)
            gt = getPreds_float(label_1,opt.hmGauss)
            for k=1,batch do
                local sum=0;
                local interocular_distance = torch.norm(gt[{{k},{37},{}}]-gt[{{k},{46},{}}])
                for j=1,pts_num do
                   sum = sum+torch.norm(gt[{{k},{j},{}}]-preds[{{k},{j},{}}])
                end
                acc = sum/(pts_num*interocular_distance)/batch;
            end
            if acc<0.05 then
                  RMSE=RMSE+1
            end
            local matio = require "matio"
	        preds = torch.reshape(preds, pts_num, 2)
		    output_ = torch.reshape(output_1, pts_num,opt.outputRes,opt.outputRes)
	        input_   = torch.reshape(input, 3,opt.inputRes,opt.inputRes)
            im_show = drawLandmarks(input_, output_, preds*(256/opt.outputRes), pts_num)
	        --image.save('./results/300VW_out_' .. indices[1] .. '.jpg', im_show)
	        --matio.save('./results/300VW_out_' .. indices[1] .. '.mat',preds:float())

        end
       
        dists=dists_1
        avgAcc = avgAcc + acc / nIters
    end

    if ref.log[tag] then
        table.insert(opt.acc[tag], avgAcc)
        ref.log[tag]:add{
            ['epoch     '] = string.format("%d",  epoch),
            ['acc       '] = string.format("%.4f" , avgAcc),
            ['LR        '] = string.format("%g" , optimState.learningRate)
        }
    end
    --print(avgAcc)
    if  tag == 'valid' then
        print('test -- ')
        print(RMSE)
    end

    if (tag == 'valid' and opt.snapshot ~= 0 and epoch % opt.snapshot == 0) or tag == 'predict' then
        -- Take a snapshot
        model:clearState()
        torch.save(paths.concat(opt.save, 'options.t7'), opt)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
        torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
	--[[
        local predFilename = 'preds.h5'
        if tag == 'predict' then predFilename = 'final_' .. predFilename end
        local predFile = hdf5.open(paths.concat(opt.save,predFilename),'w')
        for k,v in pairs(saved) do predFile:write(k,v) end
        predFile:close()
	]]--
    end
end

function train() step('train') end
function valid() step('valid') end
function predict() step('predict') end

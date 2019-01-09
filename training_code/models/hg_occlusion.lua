paths.dofile('layers/Residual.lua')

local function hourglass(n, f, inp)
    -- Upper branch
    local up1 = inp
    for i = 1,opt.nModules do up1 = Residual(f,f)(up1) end

    -- Lower branch
    local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    for i = 1,opt.nModules do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglass(n-1,f,low1)
    else
        low2 = low1
        for i = 1,opt.nModules do low2 = Residual(f,f)(low2) end
    end

    local low3 = low2
    for i = 1,opt.nModules do low3 = Residual(f,f)(low3) end
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

function createModel()

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,opt.nFeats)(r4)

    local out = {}
    local inter = r5


        local hg = hourglass(6,opt.nFeats,inter)

        -- Residual layers at output resolution
        local ll = hg
        for j = 1,opt.nModules do ll = Residual(opt.nFeats,opt.nFeats)(ll) end
        -- Linear layer to produce first set of predictions
        ll_1 = lin(opt.nFeats,opt.nFeats,ll) 

        -- Predicted heatmaps
        local tmpOut = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(ll_1)    
        table.insert(out,tmpOut)
        local tmpOut1 = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(tmpOut)
        local ll_2 = nn.CAddTable()({ll,inter,tmpOut1})
        local occlusion_1 = hourglass(6,opt.nFeats,ll_2)
        local occlusion_2 = occlusion_1
        for j = 1,opt.nModules do occlusion_2 = Residual(opt.nFeats,opt.nFeats)(occlusion_2) end
        occlusion_2 = lin(opt.nFeats,opt.nFeats,occlusion_2)
        local occlusion_3 = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(occlusion_2)
        --local occlusion_1 = nnlib.SpatialConvolution(512,512,7,7,1,1,3,3)(ll_2)
        --local occlusion_1b = nnlib.ReLU(true)(nn.SpatialBatchNormalization(512)(occlusion_1))
        ----local r1 = Residual(64,128)(cnv1)
        --local occlusion_2 = Residual(512,512)(occlusion_1b)
        --local occlusion_3 = Residual(512,512)(occlusion_2)
        --local occlusion_4 = Residual(512,opt.nFeats)(occlusion_3)
        --local occlusion_5 = lin(opt.nFeats,opt.nFeats,occlusion_4)
        --local occlusion_out = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(occlusion_5)
        --local occlusion_6 = nn.SoftMax()(occlusion_5)
        table.insert(out,occlusion_3)

        -- Add predictions back
        
        local ll_ = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll)
        local tmpOut_ = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(tmpOut)
        local occlusionOut_ = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(occlusion_3)
        inter1 = nn.CAddTable()({inter, ll_, tmpOut_,occlusionOut_})
        
        local hg1 = hourglass(6,opt.nFeats,inter1)

        -- Residual layers at output resolution
        local ll1 = hg1
        for j = 1,opt.nModules do ll1 = Residual(opt.nFeats,opt.nFeats)(ll1) end
        -- Linear layer to produce first set of predictions
        ll1 = lin(opt.nFeats,opt.nFeats,ll1)

        -- Predicted heatmaps
        local ll_2 = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll1)
        local tmpOut2 = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(ll1)
        local tmpOut2_ = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(tmpOut2)
        local occlusion_4 = nn.CAddTable()({ll_2,inter,tmpOut2_})
        local occlusion_5 = hourglass(6,opt.nFeats,occlusion_4)
        local occlusion_6 = occlusion_5
        for j = 1,opt.nModules do occlusion_6 = Residual(opt.nFeats,opt.nFeats)(occlusion_6) end
        occlusion_6 = lin(opt.nFeats,opt.nFeats,occlusion_6)
        local occlusion_7 = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(occlusion_6)
        table.insert(out,occlusion_7)
        --local ll_22 = nn.CAddTable()({occlusion_out,tmpOut1})
        --local tmpOut1 = nnlib.SpatialConvolution(ref.nOutChannels,ref.nOutChannels,1,1,1,1,0,0)(tmpOut1)
        local finaloutput = nn.CAddTable()({tmpOut2,occlusion_7})
        table.insert(out,finaloutput)
        --local occlusion_11 = nnlib.SpatialConvolution(512,512,7,7,1,1,3,3)(ll_22)
        --local occlusion_11b = nnlib.ReLU(true)(nn.SpatialBatchNormalization(512)(occlusion_11))
        ----local r1 = Residual(64,128)(cnv1)
        --local occlusion_22 = Residual(512,512)(occlusion_11b)
        --local occlusion_33 = Residual(512,512)(occlusion_22)
        --local occlusion_44 = Residual(512,opt.nFeats)(occlusion_33)
        --local occlusion_55 = lin(opt.nFeats,opt.nFeats,occlusion_44)
        --local occlusion_out1 = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(occlusion_55)
        --local occlusion_6 = nn.SoftMax()(occlusion_5)
       -- table.insert(out,occlusion_out1)
        --local occlusion_6 = nn.SoftMax()(occlusion_5)
        --table.insert(out,occlusion_66)



    -- Final model
    local netG = nn.gModule({inp}, out)

    return netG

end

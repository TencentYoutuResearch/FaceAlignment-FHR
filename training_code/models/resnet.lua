paths.dofile('layers/Residual.lua')


function createModel()

    local out = {}
    local inp = nn.Identity()()
    local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- N*64*128
    local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local pool_ = nnlib.SpatialMaxPooling(2,2,2,2)(cnv1)                    -- 64*64

    local r2a = Residual(64,256)(pool_) - nnlib.ReLU(true)                  -- 256*64
    local r2b = Residual(256,256)(r2a) - nnlib.ReLU(true)
    local r2c = Residual(256,256)(r2b) - nnlib.ReLU(true)
    
    local r3a = Residual_s(256,512,2,0)(r2c)- nnlib.ReLU(true)              -- 512*32
    local r3b = Residual(512,512)(r3a)- nnlib.ReLU(true)
    local r3c = Residual(512,512)(r3b)- nnlib.ReLU(true)
    local r3d = Residual(512,512)(r3c)- nnlib.ReLU(true)

    local r4a = Residual_s(512,1024,2,0)(r3d)- nnlib.ReLU(true)             -- 1024*16
    local r4b = Residual(1024,1024)(r4a)- nnlib.ReLU(true)
    local r4c = Residual(1024,1024)(r4b)- nnlib.ReLU(true)
    local r4d = Residual(1024,1024)(r4c)- nnlib.ReLU(true)
    local r4e = Residual(1024,1024)(r4d)- nnlib.ReLU(true)
    local r4f = Residual(1024,1024)(r4e)- nnlib.ReLU(true)


    local r5a = Residual_s(1024,2048,2,0)(r4f) - nnlib.ReLU(true)            -- 2048*8
    local r5b = Residual(2048,2048)(r5a) - nnlib.ReLU(true)
    local r5c = Residual(2048,2048)(r5b) - nnlib.ReLU(true)

    local pool_5_ = nnlib.SpatialAveragePooling(8,8,1,1)(r5c)               -- 2048*1
    --local fc_ = pool_5_ - nn.View(2048):setNumInputDims(3)
    local fc_ = pool_5_ - nn.Reshape(2048)
    local output_ = fc_ - nn.Linear(2048, 86*2) - nn.Sigmoid() 
    
    table.insert(out,output_)
    -- Final model
    local model = nn.gModule({inp}, out)

    return model

end

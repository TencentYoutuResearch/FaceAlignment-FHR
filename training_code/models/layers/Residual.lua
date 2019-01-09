local conv = nnlib.SpatialConvolution
local batchnorm = nn.SpatialBatchNormalization
local relu = nnlib.ReLU

-- Main convolutional block
local function convBlock(numIn,numOut)
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu(true))
        :add(conv(numIn,numOut/4,1,1))
        :add(batchnorm(numOut/4))
        :add(relu(true))
        :add(conv(numOut/4,numOut/4,3,3,1,1,1,1))
        :add(batchnorm(numOut/4))
        :add(relu(true))
        :add(conv(numOut/4,numOut,1,1))
end

-- Skip layer
local function skipLayer(numIn,numOut)
    if numIn == numOut then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(conv(numIn,numOut,1,1))
    end
end

-- Residual block
function Residual(numIn,numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
end

local function convBlock_s(numIn,numOut,stride_,pad_)
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu(true))
        :add(conv(numIn,numOut/4,1,1,stride_,stride_,pad_,pad_))
        :add(batchnorm(numOut/4))
        :add(relu(true))
        :add(conv(numOut/4,numOut/4,3,3,1,1,1,1))
        :add(batchnorm(numOut/4))
        :add(relu(true))
        :add(conv(numOut/4,numOut,1,1))
end

-- Skip layer
local function skipLayer_s(numIn,numOut,stride_,pad_)
    if numIn == numOut then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(conv(numIn,numOut,1,1,stride_,stride_,pad_,pad_))
    end
end

-- Residual block
function Residual_s(numIn,numOut,stride_,pad_)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock_s(numIn,numOut,stride_,pad_))
            :add(skipLayer_s(numIn,numOut,stride_,pad_)))
        :add(nn.CAddTable(true))
end


--- Load up network model or initialize from scratch
--paths.dofile('models/unet_occlusion.lua')   --hg network
paths.dofile('models/hg_128_modifiedUnit.lua') 
util = paths.dofile('util/util.lua')
input_nc = opt.input_nc
output_nc = opt.output_nc
ndf =opt.ndf
paths.dofile('models/netD.lua') 
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end
-- Continuing an experiment where it left off
if opt.continue or opt.branch ~= 'none' then
    local prevModel_G = opt.load .. '/final_model_G.t7'
    print('==> Loading model from: ' .. prevModel_G)
    netG = torch.load(prevModel_G)
    
    local prevModel_D = opt.load .. '/final_model_D.t7'
    print('==> Loading model from: ' .. prevModel_D)
    netD = torch.load(prevModel_D)
-- Or a path to previously trained model is provided
--elseif opt.loadModel_G ~= 'none' then
--    assert(paths.filep(opt.loadModel_G_G), 'File not found: ' .. opt.loadModel_G_G)
--    print('==> Loading model from: ' .. opt.loadModel_G_G)
--    netG = torch.load(opt.loadModel_G_G)
--    
--    assert(paths.filep(opt.loadModel_G_D), 'File not found: ' .. opt.loadModel_G_D)
--    print('==> Loading model from: ' .. opt.loadModel_G_D)
--    netD = torch.load(opt.loadModel_G_D)
-- Or we're starting fresh
else
    print('==> Creating model from file: models/' .. opt.netType .. '.lua')
    
    model = createModel()
   -- model = torch.load('model_70.t7')
    model:apply(weights_init)
    input_nc_tmp = input_nc  
    netD = defineD_basic(input_nc_tmp, output_nc, ndf) 
   netD:apply(weights_init)
end

-- Criterion (can be set in the opt.task file as well)
if opt.GPU ~= -1 then
    -- Convert model to CUDA
    print('==> Converting model to CUDA')
    --netD:cuda()
    model:cuda()
    criterion:cuda()
    cudnn.fastest = true
    cudnn.benchmark = true
end

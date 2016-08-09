function createModel(nGPU)
   -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
   -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
   local features = nn.Sequential()
   
   --local SpatialConvolution = nn.SpatialConvolution
   local SpatialConvolution = nn.SpatialConvolutionMKLDNN
   --local ReLU = nn.ReLU
   local ReLU = nn.ReLUMKLDNN
   local SpatialMaxPooling = nn.SpatialMaxPoolingMKLDNN
   local SpatialBatchNormalization = nn.SpatialBatchNormalizationMKLDNN

   features:add(SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
   features:add(SpatialBatchNormalization(64,1e-3))
   features:add(ReLU(true))
   features:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   features:add(SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
   features:add(SpatialBatchNormalization(192,1e-3))
   features:add(ReLU(true))
   features:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   features:add(SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
   features:add(SpatialBatchNormalization(384,1e-3))
   features:add(ReLU(true))
   features:add(SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(SpatialBatchNormalization(256,1e-3))
   features:add(ReLU(true))
   features:add(SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(SpatialBatchNormalization(256,1e-3))
   features:add(ReLU(true))
   features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

   features:get(1).gradInput = nil

   --features = makeDataParallel(features, nGPU) -- defined in util.lua

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))

   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.BatchNormalization(4096, 1e-3))
   classifier:add(nn.ReLU())

   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.BatchNormalization(4096, 1e-3))
   classifier:add(nn.ReLU())

   classifier:add(nn.Linear(4096, nClasses))
   classifier:add(nn.LogSoftMax())

   

   local model = nn.Sequential():add(features):add(classifier)
   model.imageSize = 256
   model.imageCrop = 224

   return model
end

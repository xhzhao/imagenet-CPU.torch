--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
--require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'sys'

--sys.compare = true
--sys.timerEnable = true


--sys.compare = true
--sys.timerEnable = true

sys.totalTime = 0
sys.convTime_forward = 0
sys.convTime_backward = 0
sys.maxpoolingTime_forward = 0
sys.maxpoolingTime_backward = 0
sys.avgpoolingTime_forward = 0
sys.avgpoolingTime_backward = 0
sys.reluTime_forward = 0
sys.reluTime_backward = 0
sys.lrnTime_forward = 0
sys.lrnTime_backward = 0
sys.sbnTime = 0
sys.sbnTime_forward = 0
sys.sbnTime_backward = 0
sys.linearTime_forward = 0
sys.linearTime_backward = 0
sys.dropTime_forward = 0
sys.dropTime_backward = 0
sys.concatTableTime_forward = 0
sys.concatTableTime_backward = 0
sys.concatTime_forward = 0
sys.concatTime_backward = 0
sys.thresholdTime_forward = 0
sys.thresholdTime_backward = 0



torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts_alexnet.lua')

opt = opts.parse(arg)

nClasses = opt.nClasses

paths.dofile('util.lua')
paths.dofile('model.lua')
opt.imageSize = model.imageSize or opt.imageSize
opt.imageCrop = model.imageCrop or opt.imageCrop

print(opt)

--cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('data.lua')
paths.dofile('train_alexnet.lua')
paths.dofile('test.lua')

epoch = opt.epochNumber

for i=1,opt.nEpochs do
   train()
   test()
   epoch = epoch + 1
end

--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    --weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     20,   1e-2,   5e-4, },
        { 21,     40,   1e-3,   5e-4  },
        { 41,     60,   1e-4,   5e-4 },
        { 61,     80,   1e-5,   5e-4 },
        { 80,     90,   1e-6,   5e-4 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, loss_epoch,top5_epoch
local showErrorRateInteval


-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   --TRICK  - optim input requirements
   --   --   lr  - base learning rate
   --      --   lrs - learning rate scale
   --         --   wd  - skip if wds provided
   --            --   wds - base weight decay * scale
   --

   local lrs, wds = model:getOptimConfig(1, opt.weightDecay)

   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         --weightDecay = params.weightDecay  --should be skipped
         learningRates = lrs,
         weightDecays = wds
      }
   end
   batchNumber = 0
 --  cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   top1_epoch = 0
   top5_epoch = 0
   loss_epoch = 0
   showErrorRateInteval = 100
   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels = trainLoader:sample(opt.batchSize)
            return inputs, labels
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
--[[
    if (i%1000) == 0 then
       test()
    end
]]--
   end

   donkeys:synchronize()
--   cutorch.synchronize()
--[[
   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1_epoch))
   print('\n')
]]--
   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   model:clearState()
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.Tensor()
local labels = torch.Tensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
--   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)
   --inputs:resize(inputsCPU:size())
   --labels:resize(labelsCPU:size())

   local err, outputs
   feval = function(x)
      model:zeroGradParameters()
      outputs = model:forward(inputs)
      err = criterion:forward(outputs, labels)
      local gradOutputs = criterion:backward(outputs, labels)
      model:backward(inputs, gradOutputs)
      return err, gradParameters
   end
   --adamState = {learningRate = 0.001}
   --optim.adam(feval, parameters, adamState)
   optim.sgd(feval, parameters, optimState)


   -- DataParallelTable's syncParameters
   if model.needsSync then
      model:syncParameters()
   end
   sys.initOk = 1
   if sys and sys.timerEnable then
        print("sys.totalTime =          ",sys.totalTime)
        print("sys.convTime_forward =           ",sys.convTime_forward)
        print("sys.convTime_backward =          ",sys.convTime_backward)
        print("sys.maxpoolingTime_forward =     ",sys.maxpoolingTime_forward)
        print("sys.maxpoolingTime_backward =    ",sys.maxpoolingTime_backward)
        print("sys.avgpoolingTime_forward =     ",sys.avgpoolingTime_forward)
        print("sys.avgpoolingTime_backward =    ",sys.avgpoolingTime_backward)
        print("sys.reluTime_forward =           ",sys.reluTime_forward)
        print("sys.reluTime_backward =          ",sys.reluTime_backward)
        print("sys.lrnTime_forward =            ",sys.lrnTime_forward)
        print("sys.lrnTime_backward =           ",sys.lrnTime_backward)
        print("sys.sbnTime_forward =            ",sys.sbnTime_forward)
        print("sys.sbnTime_backward =           ",sys.sbnTime_backward)
        print("sys.linearTime_forward = ",      sys.linearTime_forward)
        print("sys.linearTime_backward =        ",      sys.linearTime_backward)
        print("sys.dropTime_forward=            ",sys.dropTime_forward)
        print("sys.dropTime_backward=           ",sys.dropTime_backward)
        print("sys.concatTableTime_forward=             ",sys.concatTableTime_forward)
        print("sys.concatTableTime_backward=            ",sys.concatTableTime_backward)
        print("sys.concatTime_forward =         ",sys.concatTime_forward)
        print("sys.concatTime_backward=         ",sys.concatTime_backward)
        print("sys.thresholdTime_forward =      ",sys.thresholdTime_forward)
        print("sys.thresholdTime_backward =      ",sys.thresholdTime_backward)
        print("sys.logsoftmaxTime_forward =      ",sys.logsoftmaxTime_forward)
        print("sys.logsoftmaxTime_backward =      ",sys.logsoftmaxTime_backward)
        print("sum =                    ",sys.convTime_forward+sys.convTime_backward+sys.maxpoolingTime_forward+sys.maxpoolingTime_backward+sys.avgpoolingTime_forward+sys.avgpoolingTime_backward+sys.reluTime_forward+sys.reluTime_backward+sys.sbnTime_forward+sys.sbnTime_backward+sys.linearTime_forward+sys.linearTime_backward+sys.dropTime_forward+sys.dropTime_backward+sys.concatTime_forward+sys.concatTime_backward+sys.concatTableTime_forward+sys.concatTableTime_backward+sys.thresholdTime_forward+sys.thresholdTime_backward+sys.lrnTime_forward+sys.lrnTime_backward+sys.logsoftmaxTime_forward+sys.logsoftmaxTime_backward)
        print("------")

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
	sys.logsoftmaxTime_forward = 0
        sys.logsoftmaxTime_backward = 0
   end


   

--   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   -- top-1 error
--[[
   local top1 = 0
   do
      local _,prediction_sorted = outputs:float():sort(2, true) -- descending
      for i=1,opt.batchSize do
	 if prediction_sorted[i][1] == labelsCPU[i] then
	    top1_epoch = top1_epoch + 1;
	    top1 = top1 + 1
	 end
      end
      top1 = top1 * 100 / opt.batchSize;
   end

   local top5 = 0
   do
      local _,prediction_sorted = outputs:float():sort(2, true) -- descending
      for i=1,opt.batchSize do
        if (prediction_sorted[i][1] == labelsCPU[i] or prediction_sorted[i][2] == labelsCPU[i] or prediction_sorted[i][3] == labelsCPU[i] or prediction_sorted[i][4] == labelsCPU[i] or prediction_sorted[i][5] == labelsCPU[i] ) then
            top5_epoch = top5_epoch + 1;
            top5 = top5 + 1
        end
      end
      top5 = top5 * 100 / opt.batchSize;
   end
]]--

   -- Calculate top-1 error, and print information
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f   LR %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, err,
          optimState.learningRate, dataLoadingTime))

   dataTimer:reset()
end


function showErrorRate()

   top1_epoch = top1_epoch * 100 / (opt.batchSize * showErrorRateInteval)
   top5_epoch = top5_epoch * 100 / (opt.batchSize * showErrorRateInteval)
   loss_epoch = loss_epoch / showErrorRateInteval

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['% top5 accuracy (train set)'] = top5_epoch,
      ['avg loss (train set)'] = loss_epoch
   }   
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t top-5 %.2f \t', 
                       epoch, timer:time().real, loss_epoch, top1_epoch, top5_epoch))
   print('\n')
    
end

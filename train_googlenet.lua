--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
require 'nnlr'

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
    --dampening = 0.0,
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
        {  1,      8,    1e-2,     2e-4  },
        {  9,     16,    0.0096,   2e-4, },
        { 17,     24,    0.009216,  2e-4},
        { 25,     32,    0.00884736,   2e-4  },
        { 33,     40,    0.008493466,   2e-4 },
        { 41,     48,    0.008153727,   2e-4 },
        { 49,     56,    0.007827578,   2e-4 },
        { 57,     64,    0.007514475,   2e-4 },
        { 65,     72,    0.007213896,   2e-4 },
        { 73,     80,    0.00692534,   2e-4 },
        { 81,     88,    0.006648326,   2e-4 },
        { 89,     96,    0.006382393,   2e-4 },
        { 97,     104,   0.006127098,   2e-4 },
        { 105,    112,   0.005882014,   2e-4 },
        { 113,    120,   0.005646733,   2e-4 },
        { 121,    128,   0.005420864,   2e-4 },
        { 129,    136,   0.005204029,   2e-4 },
        { 137,    144,   0.004995868,   2e-4 },
        { 145,    152,   0.004796033,   2e-4 },
        { 153,    160,   0.004604192,   2e-4 },
        { 161,    168,   0.004420024,   2e-4 },
        { 169,    176,   0.004243223,   2e-4 },
        { 177,    184,   0.004073494,   2e-4 },
        { 185,    192,   0.003910555,   2e-4 },
        { 193,    200,   0.003754132,   2e-4 },
        { 201,    208,   0.003603967,   2e-4 },
        { 209,    216,   0.003459808,   2e-4 },
        { 217,    224,   0.003321416,   2e-4 },
        { 225,    232,   0.003188559,   2e-4 },
        { 233,    240,   0.003061017,   2e-4 },
        { 241,    248,   0.002938576,   2e-4 },
        { 249,    250,   0.002821033,   2e-4 },
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

   local params, newRegime = paramsForEpoch(epoch)
   local baseLR = params.learningRate
   local baseWD = params.weightDecay
   local LRs, WDs = model:getOptimConfig(1, baseWD)



   if newRegime then
        local bUseNNlr = true
        if bUseNNlr then
                optimState = {
                         learningRate = baseLR,
                         learningRateDecay = 0.0,
                         momentum = opt.momentum,
                         weightDecays = WDs,
                         learningRates = LRs,
                        }
        else
                optimState = {
                         learningRate = baseLR,
                         learningRateDecay = 0.0,
                         momentum = opt.momentum,
                         dampening = 0.0,
                         weightDecay = baseWD,
                        }
        end

   end
   batchNumber = 0
 --  cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()
   model.imageSize = 256
   model.imageCrop = 224
   model.auxClassifiers = 2
   model.auxWeights = {0.3, 0.3}



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

   local err, outputs, totalerr
   feval = function(x)
      model:zeroGradParameters()
      --outputs = model:forward(inputs)
      --err = criterion:forward(outputs, labels)
      --local gradOutputs = criterion:backward(outputs, labels)
      --model:backward(inputs, gradOutputs)
      --return err, gradParameters

      outputs = model:forward(inputs)
      local model_outputs = outputs:sub(1, -1, 1, nClasses)
      err = criterion:forward(model_outputs, labels)
      totalerr = err
      local gradOutputs = criterion:backward(model_outputs, labels)

      if model.auxClassifiers and model.auxClassifiers > 0 then
         local allGradOutputs = torch.Tensor():typeAs(gradOutputs):resizeAs(outputs)
         allGradOutputs:sub(1, -1, 1, nClasses):copy(gradOutputs)
         auxerr = {}
         for i=1,model.auxClassifiers do
            local first = i * nClasses + 1
            local last = (i+1) * nClasses
            local classifier_outputs = outputs:sub(1, -1, first, last)
            auxerr[i] = criterion:forward(classifier_outputs, labels)
            totalerr = totalerr + auxerr[i] * model.auxWeights[i]
            local auxGradOutput = criterion:backward(classifier_outputs, labels) * model.auxWeights[i]
            allGradOutputs:sub(1, -1, first, last):copy(auxGradOutput)
         end
         gradOutputs = allGradOutputs
      end
      model:backward(inputs, gradOutputs)
      return totalerr, gradParameters

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
--[[
   -- top-1 error
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
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f  LR %.8e DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, totalerr,
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

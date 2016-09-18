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
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

if opt.rngState ~= 'none' then
    assert(paths.filep(opt.rngState), 'File not found: ' .. opt.rngState)
    print('Loading RNG state from file: ' .. opt.rngState)
    loadRNGState(opt.rngState, donkeys, opt.nDonkeys)
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
local top1_epoch, loss_epoch
trainConf = opt.conf and optim.ConfusionMatrix(classes) or nil

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()
   local baseLearningRate = optimState.learningRate
   local baseWeightDecay = optimState.weightDecay
   local learningRates, weightDecays = model:getOptimConfig(baseLearningRate, baseWeightDecay)
--[[
   print("start to print learningRates,weightDecays")
   print(#learningRates)
   print(#weightDecays)
   for i=1,100 do
      print(learningRates[i])
   end
]]--
   --print( weightDecays)
   optimState['learningRates'] = learningRates
   optimState['weightDecays'] = weightDecays
   print(optimState)


   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0
   if trainConf then trainConf:zero() end
   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels
            local ok = xpcall(function()
                                 inputs, labels = trainLoader:sample(opt.batchSize)
                              end, function()
                                 print("ERROR!")
                                 print(debug.traceback())
                              end);
            if not ok then
               return
            end
            -- check the error
            return inputs, labels
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

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

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   local function sanitize(net)
      local list = net:listModules()
      for _,val in ipairs(list) do
         for name,field in pairs(val) do
            if torch.type(field) == 'cdata' then val[name] = nil end
            if (name == 'output' or name == 'gradInput') then
               if torch.isTensor(val[name]) then
                  val[name] = field.new()
               elseif torch.type(val[name]) == 'table' then
                  val[name] = {}
               end
            end
         end
      end
   end
   sanitize(model)
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   saveRNGState(paths.concat(opt.save, 'rngState_' .. epoch .. '.t7'), donkeys, opt.nDonkeys) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
   if not inputsCPU then
      print("Loader error. Skipping batch.")
      return
   end

   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   local outputsCPU = torch.FloatTensor(opt.batchSize, nClasses)

   local err, auxerr, totalerr, outputs
   feval = function(x)
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
      -- This division could go into chunkedFeval but is in here for performance reasons
      gradOutputs:mul(1.0 / opt.batchChunks)

      model:backward(inputs, gradOutputs)
      return totalerr, gradParameters
   end

   local chunkedInputsCPU, chunkedLabelsCPU
   local transferToGPU = function(from, to)
      chunkedInputsCPU = inputsCPU:sub(from, to)
      chunkedLabelsCPU = labelsCPU:sub(from, to)
      inputs:resize(chunkedInputsCPU:size()):copy(chunkedInputsCPU)
      labels:resize(chunkedLabelsCPU:size()):copy(chunkedLabelsCPU)
   end

   -- This simulates feval by dividing every batch into chunks and calling
   -- the original function
   local chunkedFeval = function(x)
      local chunk_size = math.floor(opt.batchSize / opt.batchChunks)
      local err_accumulator = 0
      for i=1,opt.batchChunks do
         local chunk_start = chunk_size * (i-1) + 1
         -- Take all remaining samples in the last iteration
         local chunk_end = i < opt.batchChunks and chunk_size * i or -1
         transferToGPU(chunk_start, chunk_end)
         local loss, _ = feval(x)
         err_accumulator = err_accumulator + loss
         outputsCPU:sub(chunk_start, chunk_end):copy(outputs:sub(1, -1, 1, nClasses))
      end
      totalerr = err_accumulator / opt.batchChunks
      return total_loss, gradParameters
   end

   model:zeroGradParameters()


   optim.sgd(chunkedFeval, parameters, optimState)

   -- DataParallelTable's syncParameters
   model:apply(function(m) if m.syncParameters then m:syncParameters() end end)

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + totalerr
   -- top-1 error
   local top1 = 0
   do
      local _,max_prediction = outputsCPU:max(2)
      for i=1,opt.batchSize do
         if max_prediction[i][1] == labelsCPU[i] then
            top1_epoch = top1_epoch + 1;
            top1 = top1 + 1
         end
      end
      top1 = top1 * 100 / opt.batchSize;
   end

   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, totalerr, top1,
          optimState.learningRate, dataLoadingTime))

   if model.auxClassifiers and model.auxClassifiers > 0 then
      print(string.format('\t  main model: Err %.4f', err))
      for i=1,model.auxClassifiers do
         print(string.format('\tclassifier %d: Err %.4f (* %.1f = %.4f)', i, auxerr[i], model.auxWeights[i], auxerr[i] * model.auxWeights[i]))
      end
   end

   if trainConf then trainConf:batchAdd(outputsCPU, labelsCPU) end

   dataTimer:reset()
end

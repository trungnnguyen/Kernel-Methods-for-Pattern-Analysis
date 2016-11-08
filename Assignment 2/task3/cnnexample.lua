require 'paths';
require 'nn';
lmat = require 'matio';

-- if (not paths.filep("cifar10torchsmall.zip")) then
--     os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
--     os.execute('unzip cifar10torchsmall.zip')
-- end

alldata = lmat.load('cnninput.mat');
trainset = alldata.trainset
valset = alldata.valset
testset = alldata.testset

-- trainset.data = trainset.data:byte()
-- testset.data = testset.data:byte()
-- trainset.label = trainset.label:byte()
-- testset.label = testset.label:byte()

-- change testset to valset for validation


-- dataset={};
-- function dataset:size() return 2090 end -- 100 examples
-- for i=1,dataset:size() do 
--   dataset[i] = {trainset.data[i]:double(), trainset.label[i]:double()}
-- end
-- print(dataset[1])
-- trainset.data = alldata.trainset.data
-- testset.data = alldata.testset.data
-- trainset.label = torch.Bytetensor(2090)
-- testset.label = torch.Bytetensor(298)

for i=1,2090 do
	trainset.label[i] = alldata.trainset.label[i]
end

for i=1,298 do
	testset.label[i] = alldata.testset.label[i]
end


-- trainset = alldata.trainset
-- testset = alldata.testset



-- trainset = torch.load('cifar10-train.t7')
-- testset = torch.load('cifar10-test.t7')

classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

print(trainset)
print(#trainset.data)

--itorch.image(trainset.data[100]) -- display the 100-th image in dataset
print(classes[trainset.label[100]])

-- ignore setmetatable for now, it is a feature beyond the scope of this tutorial. It sets the index operator.
-- setmetatable(testset, 
--    {__index = function(t, i) 
--                    return {t.data[i], t.label[i]} 
--                end}
-- );
testset.data = testset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.


setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.


function trainset:size() 
    return self.data:size(1) 
end

print(trainset:size()) -- just to test
print(trainset[33]) -- load sample number 33.
--itorch.image(trainset[33][1])

redChannel = trainset.data[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}

print(#redChannel)

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- net = nn.Sequential()
-- net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
-- net:add(nn.ReLU())                       -- non-linearity 
-- net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
-- net:add(nn.SpatialConvolution(6, 16, 5, 5))
-- net:add(nn.ReLU())                       -- non-linearity 
-- net:add(nn.SpatialMaxPooling(2,2,2,2))
-- net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
-- net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
-- net:add(nn.ReLU())                       -- non-linearity 
-- net:add(nn.Linear(120, 84))
-- net:add(nn.ReLU())                       -- non-linearity 
-- net:add(nn.Linear(84, 5))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
-- net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems


net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 3, 3)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.Tanh())                       -- non-linearity 
net:add(nn.SpatialAveragePooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 3, 3)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.Tanh())                       -- non-linearity 
net:add(nn.SpatialAveragePooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(16, 64, 3, 3)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.Tanh())                       -- non-linearity 
net:add(nn.SpatialAveragePooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.View(64*2*2))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(64*2*2, 100))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.Tanh())                       -- non-linearity 
net:add(nn.Linear(100, 56))
net:add(nn.Tanh())                       -- non-linearity 
net:add(nn.Linear(56, 5))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems



-- print(net);

-- criterion = nn.CrossEntropyCriterion()
criterion = nn.ClassNLLCriterion()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.0008
trainer.maxIteration = 120 -- just do 5 epochs of training.

trainer:train(trainset)

print(classes[testset.label[100]])
--itorch.image(testset.data[100])

testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- for fun, print the mean and standard-deviation of example-100
horse = testset.data[100]
print(horse:mean(), horse:std())

print(classes[testset.label[100]])
--itorch.image(testset.data[100])
predicted = net:forward(testset.data[100])

-- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 
print(predicted:exp())

for i=1,predicted:size(1) do
    print(classes[i], predicted[i])
end

correct = 0
predlabel = torch.Tensor(298,2)
for i=1,298 do
	local groundtruth = testset.label[i][1]	
    -- local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
	 -- print(prediction)
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
	--print(indices)
	predlabel[i][1]=groundtruth
	predlabel[i][2]=indices[1]
	print(groundtruth, indices[1])
    if groundtruth == indices[1] then
	-- print('inside')	
        correct = correct + 1
    end
end

print(correct, 100*correct/298 .. ' % ')
lmat.save('cnnout120and100feat64.mat',predlabel)





net:remove()
net:remove()
net:remove()
net:remove()
net:remove()

featstrain= torch.Tensor(2090,100)
for i=1,2090 do
	local prediction = net:forward(trainset.data[i])
	featstrain[i] = prediction
end
lmat.save('cnnoutfeatstrain120and100feat64.mat',featstrain)

featstest= torch.Tensor(298,100)
for i=1,298 do
	local prediction = net:forward(testset.data[i])
	featstest[i] = prediction
end
lmat.save('cnnoutfeatstest120and100feat64.mat',featstest)


-- class_performance = {0, 0, 0, 0, 0}
-- for i=1,10000 do
--     local groundtruth = testset.label[i]
--     local prediction = net:forward(testset.data[i])
--     local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
--     if groundtruth == indices[1] then
--         class_performance[groundtruth] = class_performance[groundtruth] + 1
--     end
-- end

-- for i=1,#classes do
--     print(classes[i], 100*class_performance[i]/1000 .. ' %')
-- end



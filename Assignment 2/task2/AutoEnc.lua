
require 'nn';
lmat = require 'matio';
Tra = lmat.load('TotalTrain.mat');

featureTrain = Tra.TrainData;

Tes = lmat.load('TotalTest.mat')
featuresTes = Tes.TestData;

Val = lmat.load('TotalVal.mat');
featureVal = Val.ValData;

hid1 = 800;
botNeck = 50;
hid2 = 800;
featureTrain = featureTrain:double()
featureVal = featureVal:double()
featuresTes = featuresTes:double()

--creating a feedforward network
Model = nn.Sequential();

--Input 
Model:add(nn.Reshape(512));

--first hidden layer
firstLayer=nn.Linear(512,hid1);
Model:add(firstLayer);
Model:add(nn.Tanh());

--second or bottleneck feature layer
secondLayer=nn.Linear(hid1,botNeck);
Model:add(secondLayer);

--third hidden layer
thirdLayer=nn.Linear(botNeck,hid2);
Model:add(thirdLayer);
Model:add(nn.Tanh());

--fourth
fourthLayer = nn.Linear(hid2,512);
Model:add(fourthLayer);
Model:add(nn.Reshape(512));


 criterion = nn.MSECriterion()



--trainer = nn.StochasticGradient(net, criterion)
--trainer.learningRate = 0.001
--trainer.maxIteration = 5 -- just do 5 epochs of training.



--trainer:train(featureTrain);

function  gradUpdate(Model, x, y, criterion, learningRate)
   local pred = Model:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   Model:zeroGradParameters()
   Model:backward(x, gradCriterion)
   Model:updateParameters(learningRate)
return err
end

for i = 1, 1000 do
  error =  gradUpdate(Model, featureTrain,featureTrain, criterion,0.01)
   if true then
     -- o1 = mlp1:forward{x, y}[1]
     -- o2 = mlp2:forward{x, z}[1]
     -- o = crit:forward(mlpa:forward{{x, y}, {x, z}}, 1)
      print(error)
   end
end

predTrain = Model:forward(featureTrain)
newTrain = Model:get(4).output;

ex = 'Train' .. botNeck;
ex = ex .. '.mat'
lmat.save(ex,newTrain);

predVal = Model:forward(featureVal)
newVal = Model:get(4).output;

ex = 'Val' .. botNeck;
ex = ex .. '.mat'
lmat.save(ex,newVal);

predTest = Model:forward(featuresTes)
newTest = Model:get(4).output;
ex = 'Test' .. botNeck;
ex = ex .. '.mat'
lmat.save(ex,newTest);







--printit




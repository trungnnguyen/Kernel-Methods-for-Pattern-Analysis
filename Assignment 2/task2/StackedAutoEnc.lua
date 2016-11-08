
require 'nn';
lmat = require 'matio';
Tra = lmat.load('TotalTrain.mat');

featureTrain = Tra.TrainData;

Tes = lmat.load('TotalTest.mat')
featuresTes = Tes.TestData;

Val = lmat.load('TotalVal.mat');
featureVal = Val.ValData;

hid1 = 800;

hid2 = 800;

hid3 = 600

hid4 = 600

hid5 = 400
hid6 = 400

botNeck = torch.Tensor({512,400,300,120});
print(botNeck[1]);


featureTrain = featureTrain:double()


firstRedDimension = 450;
secondRedDimension = 250;
thirdRedDimension = 150;

featureVal = featureVal:double()
featuresTes = featuresTes:double()

print('creating nn ');

org = 512
 ----------- giving the first reduced dimension
--creating a feedforward network
Model = nn.Sequential();

--Input 
Model:add(nn.Reshape(org));
print('creating first layer ');
--first hidden layer
firstLayer=nn.Linear(org,hid1);
Model:add(firstLayer);
Model:add(nn.Tanh());

print('creating first hid layer ');

--second or bottleneck feature layer
secondLayer=nn.Linear(hid1,firstRedDimension);
Model:add(secondLayer);
print('creating bottleneck layer ');

--third hidden layer
thirdLayer=nn.Linear(firstRedDimension,hid2);
Model:add(thirdLayer);
Model:add(nn.Tanh());
print('creating third hid layer ');
--fourth
fourthLayer = nn.Linear(hid2,org);
Model:add(fourthLayer);
Model:add(nn.Reshape(org));

print(org .. " " .. hid1 .. " " .. firstRedDimension .. " " .. hid2 .. " " .. org);
print('Neural net created');
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
print(err);
end

for i = 1, 1000 do
print('epoch ' .. i)
  gradUpdate(Model, featureTrain,featureTrain, criterion,0.01)
   if true then
     -- o1 = mlp1:forward{x, y}[1]
     -- o2 = mlp2:forward{x, z}[1]
     -- o = crit:forward(mlpa:forward{{x, y}, {x, z}}, 1)
     -- print(error)
   end
end

print('dimension of featureTrain before giving to model');

print (#featureTrain);

predTrain = Model:forward(featureTrain)
featureTrain = Model:get(4).output;


print('dimension of featureTrain before saving to .mat');

print (#featureTrain);
ex = 'stTrain' .. firstRedDimension;
ex = ex .. '.mat'
lmat.save(ex,featureTrain);


print('dimension of featureTrain after  saving it as .mat file');

print (#featureTrain);

predVal = Model:forward(featureVal)
featureVal = Model:get(4).output;




ex = 'stVal' .. firstRedDimension;
ex = ex .. '.mat'
lmat.save(ex,featureVal);




predTest = Model:forward(featuresTes)
featuresTes = Model:get(4).output;


ex = 'stTest' .. firstRedDimension;
ex = ex .. '.mat'
lmat.save(ex,featuresTes);


----loading training data for second auto encoder

newTrain = 'stTrain' .. firstRedDimension ;
newTrain = newTrain .. '.mat'

Tra = lmat.load(newTrain);

featureTrain = Tra.x;


newVal = 'stVal' .. firstRedDimension ;
newVal = newVal .. '.mat'

Val = lmat.load(newVal);
featureVal = Val.x;

newTest = 'stTest' .. firstRedDimension ;
newTest = newTest .. '.mat'

Tes = lmat.load(newTest)
featuresTes = Tes.x;


featureTrain = featureTrain:double()




featureVal = featureVal:double()
featuresTes = featuresTes:double()

print('creating nn ');

org = firstRedDimension;
 ----------- giving the first reduced dimension
--creating a feedforward network
Model = nn.Sequential();

--Input 
Model:add(nn.Reshape(org));
print('creating first layer ');
--first hidden layer
firstLayer=nn.Linear(org,hid3);
Model:add(firstLayer);
Model:add(nn.Tanh());

print('creating first hid layer ');

--second or bottleneck feature layer
secondLayer=nn.Linear(hid3,secondRedDimension);
Model:add(secondLayer);
print('creating bottleneck layer ');

--third hidden layer
thirdLayer=nn.Linear(secondRedDimension,hid4);
Model:add(thirdLayer);
Model:add(nn.Tanh());
print('creating third hid layer ');
--fourth
fourthLayer = nn.Linear(hid4,org);
Model:add(fourthLayer);
Model:add(nn.Reshape(org));

print(org .. " " .. hid3 .. " " .. secondRedDimension .. " " .. hid4 .. " " .. org);
print('Neural net created');
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
print(err);
end

for i = 1, 1000 do
print('epoch ' .. i)
  gradUpdate(Model, featureTrain,featureTrain, criterion,0.01)
   if true then
     -- o1 = mlp1:forward{x, y}[1]
     -- o2 = mlp2:forward{x, z}[1]
     -- o = crit:forward(mlpa:forward{{x, y}, {x, z}}, 1)
     -- print(error)
   end
end

print('dimension of featureTrain before giving to model');

print (#featureTrain);

predTrain = Model:forward(featureTrain)
featureTrain = Model:get(4).output;


print('dimension of featureTrain before saving to .mat');

print (#featureTrain);
ex = 'stTrain' .. secondRedDimension;
ex = ex .. '.mat'
lmat.save(ex,featureTrain);


print('dimension of featureTrain after  saving it as .mat file');

print (#featureTrain);

predVal = Model:forward(featureVal)
featureVal = Model:get(4).output;




ex = 'stVal' .. secondRedDimension;
ex = ex .. '.mat'
lmat.save(ex,featureVal);




predTest = Model:forward(featuresTes)
featuresTes = Model:get(4).output;


ex = 'stTest' .. secondRedDimension;
ex = ex .. '.mat'
lmat.save(ex,featuresTes);



newTrain = 'stTrain' .. secondRedDimension ;
newTrain = newTrain .. '.mat'

Tra = lmat.load(newTrain);

featureTrain = Tra.x;


newVal = 'stVal' .. secondRedDimension ;
newVal = newVal .. '.mat'

Val = lmat.load(newVal);
featureVal = Val.x;

newTest = 'stTest' .. secondRedDimension ;
newTest = newTest .. '.mat'

Tes = lmat.load(newTest)
featuresTes = Tes.x;



featureTrain = featureTrain:double()




featureVal = featureVal:double()
featuresTes = featuresTes:double()

print('creating nn ');

org = secondRedDimension;
 ----------- giving the first reduced dimension
--creating a feedforward network
Model = nn.Sequential();

--Input 
Model:add(nn.Reshape(org));
print('creating first layer ');
--first hidden layer
firstLayer=nn.Linear(org,hid5);
Model:add(firstLayer);
Model:add(nn.Tanh());

print('creating first hid layer ');

--second or bottleneck feature layer
secondLayer=nn.Linear(hid5,thirdRedDimension);
Model:add(secondLayer);
print('creating bottleneck layer ');

--third hidden layer
thirdLayer=nn.Linear(thirdRedDimension,hid6);
Model:add(thirdLayer);
Model:add(nn.Tanh());
print('creating third hid layer ');
--fourth
fourthLayer = nn.Linear(hid6,org);
Model:add(fourthLayer);
Model:add(nn.Reshape(org));

print(org .. " " .. hid5 .. " " .. thirdRedDimension .. " " .. hid6 .. " " .. org);
print('Neural net created');
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
print(err);
end

for i = 1, 1000 do
print('epoch ' .. i)
  gradUpdate(Model, featureTrain,featureTrain, criterion,0.01)
   if true then
     -- o1 = mlp1:forward{x, y}[1]
     -- o2 = mlp2:forward{x, z}[1]
     -- o = crit:forward(mlpa:forward{{x, y}, {x, z}}, 1)
     -- print(error)
   end
end

print('dimension of featureTrain before giving to model');

print (#featureTrain);

predTrain = Model:forward(featureTrain)
featureTrain = Model:get(4).output;


print('dimension of featureTrain before saving to .mat');

print (#featureTrain);
ex = 'stTrain' .. thirdRedDimension;
ex = ex .. '.mat'
lmat.save(ex,featureTrain);


print('dimension of featureTrain after  saving it as .mat file');

print (#featureTrain);

predVal = Model:forward(featureVal)
featureVal = Model:get(4).output;




ex = 'stVal' .. thirdRedDimension;
ex = ex .. '.mat'
lmat.save(ex,featureVal);




predTest = Model:forward(featuresTes)
featuresTes = Model:get(4).output;


ex = 'stTest' .. thirdRedDimension;
ex = ex .. '.mat'
lmat.save(ex,featuresTes);















--printit




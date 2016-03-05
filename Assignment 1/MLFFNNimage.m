clear
clc
load('CompleteData.mat');
ThreadMill=CompleteData{4};
Mattress=CompleteData{7};
Binoculars=CompleteData{17};
Ladders=CompleteData{11};
HotTub=CompleteData{12};

% ThreadTrain=ThreadMill(1:73,:);
% TTrO=ones(73,1)*[1,0,0,0,0];
% ThreadVal=ThreadMill(74:117,:);
% TVO=ones(44,1)*[1,0,0,0,0];
% ThreadTest=ThreadMill(118:147,:);
% TTO=ones(30,1)*[1,0,0,0,0];
% MatTrain=Mattress(1:95,:);
% MTrO=ones(95,1)*[0,1,0,0,0];
% MatVal=Mattress(96:153,:);
% MVO=ones(58,1)*[0,1,0,0,0];
% MatTest=Mattress(154:191,:);
% MTO=ones(38,1)*[0,1,0,0,0];
% 
% BinTrain=Binoculars(1:108,:);
% BTrO=ones(108,1)*[0,0,1,0,0];
% BinVal=Binoculars(109:173,:);
% BVO=ones(65,1)*[0,0,1,0,0];
% BinTest=Binoculars(174:216,:);
% BTO=ones(43,1)*[0,0,1,0,0];
% 
% LadTrain=Ladders(1:119,:);
% LTrO=ones(119,1)*[0,0,0,1,0];
% LadVal=Ladders(120:190,:);
% LVO=ones(71,1)*[0,0,0,1,0];
% LadTest=Ladders(191:238,:);
% LTO=ones(48,1)*[0,0,0,1,0];
% 
% HotTrain=HotTub(1:78,:);
% HTrO=ones(78,1)*[0,0,0,0,1];
% HotVal=HotTub(79:125,:);
% HVO=ones(47,1)*[0,0,0,0,1];
% HotTest=HotTub(126:156,:);
% HTO=ones(31,1)*[0,0,0,0,1];
% 
% Train=[ThreadTrain;MatTrain;BinTrain;LadTrain;HotTrain];
% Val=[ThreadVal;MatVal;BinVal;LadVal;HotVal];
% Test=[ThreadTest;MatTest;BinTest;LadTest;HotTest];
% totaldata=[Train;Val;Test];
% 
% TrainO=[TTrO;MTrO;BTrO;LTrO;HTrO];
% ValO=[TVO;MVO;BVO;LVO;HVO];
% TestO=[TTO;MTO;BTO;LTO;HTO];
% outputdata=[TrainO;ValO;TestO];
totaldata=[Binoculars;HotTub;Ladders;Mattress;ThreadMill];
outputdata=[ones(216,1)*[1,0,0,0,0];ones(156,1)*[0,1,0,0,0];ones(238,1)*[0,0,1,0,0];ones(191,1)*[0,0,0,1,0];ones(147,1)*[0,0,0,0,1]];

% hold on;
% set(gca,'fontsize',18)
% scatter([class1_train(:,1);class1_val(:,1);class1_test(:,1)],[class1_train(:,2);class1_val(:,2);class1_test(:,2)]);
% scatter([class2_train(:,1);class2_val(:,1);class2_test(:,1)],[class2_train(:,2);class2_val(:,2);class2_test(:,2)]);
% scatter([class3_train(:,1);class3_val(:,1);class3_test(:,1)],[class3_train(:,2);class3_val(:,2);class3_test(:,2)]);
% scatter([class4_train(:,1);class4_val(:,1);class4_test(:,1)],[class4_train(:,2);class4_val(:,2);class4_test(:,2)]);
% xlabel('x-axis');
% ylabel('y-axis');
% legend('Class 1','Class 2', 'Class 3', 'Class 4');
%nnstart

x = totaldata';
t = outputdata';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainbr';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = [20,20];
net = patternnet(hiddenLayerSize);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'divideint';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 50/100;
net.divideParam.valRatio = 30/100;
net.divideParam.testRatio = 20/100;

%net.divideParam.trainInd=[1:473];
%net.divideParam.valInd=[474:758];
%net.divideParam.testInd=[758:948];
% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'crossentropy';  % Cross-Entropy

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end



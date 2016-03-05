clear
clc
class1_test=load('group6\class1_test.txt');
class1_train=load('group6\class1_train.txt');
class1_val=load('group6\class1_val.txt');

class2_test=load('group6\class2_test.txt');
class2_train=load('group6\class2_train.txt');
class2_val=load('group6\class2_val.txt');

class3_test=load('group6\class3_test.txt');
class3_train=load('group6\class3_train.txt');
class3_val=load('group6\class3_val.txt');
totaldata=[class1_train;class1_val;class1_test;class2_train;class2_val;class2_test;class3_train;class3_val;class3_test];
outputdata=[ones(300,1)*[1,0,0];ones(600,1)*[0,1,0];ones(800,1)*[0,0,1]];

% hold on;
% set(gca,'fontsize',18)
% scatter([class1_train(:,1);class1_val(:,1);class1_test(:,1)],[class1_train(:,2);class1_val(:,2);class1_test(:,2)]);
% scatter([class2_train(:,1);class2_val(:,1);class2_test(:,1)],[class2_train(:,2);class2_val(:,2);class2_test(:,2)]);
% scatter([class3_train(:,1);class3_val(:,1);class3_test(:,1)],[class3_train(:,2);class3_val(:,2);class3_test(:,2)]);
% xlabel('x-axis');
% ylabel('y-axis');
% legend('Class 1','Class 2', 'Class 3');
%nnstart

x = totaldata';
t = outputdata';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainbscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = [4,4];
net = patternnet(hiddenLayerSize);

%%%%%%%%%%%%%%%%%%%%%%%%
net.trainParam.min_grad=1e-10;





%%%%%%%%%%%%%%%%%

net.trainParam.epochs=100;

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
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);

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
save('m4');
% 
% xMin=-4;
% xMax=4;
% 
% yMin=-4;
% yMax=4;
% 
% 
% 
% gridX = xMin : 0.1 : xMax;
% gridY = yMin : 0.1: yMax;
% [Xgrid,Ygrid] = meshgrid(gridX,gridY);
% XgridNew = reshape(Xgrid,[size(Xgrid,1)*size(Xgrid,2),1]);
% YgridNew = reshape(Ygrid,[size(Ygrid,1)*size(Ygrid,2),1]);
% testDataGrid = [XgridNew YgridNew];
% modelGrid = net(testDataGrid');
% outputde = vec2ind(modelGrid);
% [~,D]=size(modelGrid);
% % figure();
% % hold on;
% % set(gca,'fontsize',18)
% % for i=1:D
% %     if(outputde(i)==1)
% %         scatter(testDataGrid(i,1), testDataGrid(i,2),'r','filled');
% %     
% %     elseif(outputde(i)==2)
% %         scatter(testDataGrid(i,1), testDataGrid(i,2),'g','filled');
% %     
% %     elseif(outputde(i)==3)
% %         scatter(testDataGrid(i,1), testDataGrid(i,2),'b','filled');
% %     end
% % end
% % title('Decision Region');
% % xlabel('Feature 1');
% % ylabel('Feature 2');
% %legend('Class 1','Class 2', 'Class 3');
% modelGridNew1 = reshape(modelGrid(1,:),[size(Xgrid,1),size(Ygrid,1)]);
% modelGridNew2 = reshape(modelGrid(2,:),[size(Xgrid,1),size(Ygrid,1)]);
% modelGridNew3 = reshape(modelGrid(3,:),[size(Xgrid,1),size(Ygrid,1)]);
% figure;
% hold on;
% surf(Xgrid,Ygrid,modelGridNew1);
% surf(Xgrid,Ygrid,modelGridNew2);
% surf(Xgrid,Ygrid,modelGridNew3);
% %colormap winter
% 
% %hold on;
% %scatter3(totaldata(:,1),totaldata(:,2),outputdata(:,3),15,[0 1 0],'filled');
% title('output of output node');
% %colormap winter
% %contourf(Xgrid,Ygrid,modelGridNew3)
% color bar
% 

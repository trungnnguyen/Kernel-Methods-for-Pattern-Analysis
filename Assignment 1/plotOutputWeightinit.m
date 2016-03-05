clear
pathName = 'BIVARIATE';
fileName = '/group6_train.txt';
newString = strcat(pathName,fileName);
trainDataFull = load(newString);
trainData = trainDataFull(:,1:2);
trainDataTarget  = trainDataFull(:,3);
clearvars trainDataFull;

fileName = '/group6_val.txt';
newString = strcat(pathName,fileName);
valDataFull = load(newString);
valData = valDataFull(:,1:2);
valDataTarget  = valDataFull(:,3);
clearvars valDataFull;

fileName = '/group6_test.txt';
newString = strcat(pathName,fileName);
testDataFull = load(newString);
testData = testDataFull(:,1:2);
testDataTarget  = testDataFull(:,3);
clearvars testDataFull;

input = [trainData;valData;testData];
target = [trainDataTarget;valDataTarget;testDataTarget];
save input.mat
save target.mat

%disp('trainRatio');
trainRatio = (size(trainData,1))/(size(trainData,1) + size(valData,1)+size(testData,1));

%disp('valRatio');
valRatio = (size(valData,1))/(size(trainData,1) + size(valData,1)+size(testData,1));

%disp('testRatio');
testRatio = (size(testData,1))/(size(trainData,1) + size(valData,1)+size(testData,1));


x = input';
t = target';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. NFTOOL falls back to this in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt

% Create a Fitting net{i}work

hiddenLayer1 = [1,3,5,8,10,15,10,6];
hiddenLayer2 = [3,1,9,4,10,12,17,20];
net = cell(size(hiddenLayer1));
trainPerformance = zeros(size(hiddenLayer1));
valPerformance = zeros(size(hiddenLayer1));
testPerformance = zeros(size(hiddenLayer1));
minError = Inf;
minIndex = -1;

hidden1 = 3;  %[3 3] for 20
hidden2 = 3;  %[5 9]for 2000
epochs = [1 2 5 10 20 50 100 150 200 300]
initweight = rand(25,1);
performanceArr = zeros(1,9);
for ep = 1 : size(epochs,2);

    hiddenLayers = [hidden1,hidden2];
    net = feedforwardnet(hiddenLayers,trainFcn);
    
    net = configure(net,x,t);
net = setwb(net,initweight);

    % Setup Division of Data for Training, Validation, Testing
     net.divideFcn = 'divideind';
    net.divideParam.trainInd = 1 :2000;
    net.divideParam.valInd = 2001:2300;
    net.divideParam.testInd = 2301:2500;
%      net.divideFcn = 'dividerand';
%          net.divideParam.trainRatio = trainRatio;
%     net.divideParam.valInd = valRatio;
%     net.divideParam.testInd = testRatio;
     
    net.performFcn = 'mse';  % Mean squared error
    
    % Choose Plot Functions
    % For a list of all plot functions type: help nnplot
   % net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
     %   'plotregression', 'plotfit'};
    % Train the net{i}work
   
    net.trainParam.epochs = epochs(ep);
   
    [net,tr] = train(net,x,t);
    
    % Test the net{i}work
    modelOutput = net(x);
    % e = gsubtract(t,y);
    performanceArr(1,ep) = perform(net,t,modelOutput);
    modelOutTrain = modelOutput(tr.trainInd);
    modelOutVal = modelOutput(tr.valInd);
    modelOutTest = modelOutput(tr.testInd);
    
 %scatter plot and Target Vs train for test data
%        h =  figure;
% targetTrain = [trainData trainDataTarget];
% modelOutput = net(trainData');
% 
% scatter3(targetTrain(:,1),targetTrain(:,2),targetTrain(:,3),15,[0 1 0],'filled');
% hold on;
% scatter3(targetTrain(:,1),targetTrain(:,2),modelOutput',30,[1 0 0],'filled');
% title('Model output vs Target output for TRAIN data ');
% 
% legend('Target Output','Model Output');
% xlabel('Input variable x1');
% ylabel('Input variable x2');
% zlabel('Output');
%  saveas(h,'2000 samples TOMO Train .fig');
% 
% view([-35,35]);
% h = figure;
% plot(targetTrain(:,3),modelOutput','ob');
% hold on;
% plot((-100 : 100) ,(-100 : 100));
% xlabel('Target Output');
% ylabel('Model Output');
% zlabel('Output');
% saveas(h,'Scatter Train .fig');
% 
% %scatter plot and Target Vs train for validation data
% h =  figure;
% targetVal = [valData valDataTarget];
% modelOutput = net(valData');
% 
% scatter3(targetVal(:,1),targetVal(:,2),targetVal(:,3),15,[0 1 0],'filled');
% hold on;
% scatter3(targetVal(:,1),targetVal(:,2),modelOutput',30,[1 0 0],'filled');
% title('Model output vs Target output for VALIDATION data ');
% 
% legend('Target Output','Model Output');
% xlabel('Input variable x1');
% ylabel('Input variable x2');
% zlabel('Output');
%  saveas(h,'2000 samples TOMO Validation .fig');
% 
% view([-35,35]);
% 
% h = figure;
% plot(targetVal(:,3),modelOutput','ob');
% hold on;
% plot((-100 : 100) ,(-100 : 100));
% xlabel('Target Output');
% ylabel('Model Output');
% zlabel('Output');
% saveas(h,'Scatter Val .fig');


% %scatter plot and Target Vs train for test data
% 
%  h =  figure;
% targetTest = [testData testDataTarget];
% modelOutput = net(testData');
% 
% scatter3(targetTest(:,1),targetTest(:,2),targetTest(:,3),15,[0 1 0],'filled');
% hold on;
% scatter3(targetTest(:,1),targetTest(:,2),modelOutput',30,[1 0 0],'filled');
% title('Model output vs Target output for TEST data ');
% 
% legend('Target Output','Model Output');
% xlabel('Input variable x1');
% ylabel('Input variable x2');
% zlabel('Output');
%  saveas(h,'2000 samples TOMO Test .fig');
% view([-35,35]);
% 
% h = figure;
% plot(targetTest(:,3),modelOutput','ob');
% hold on;
% plot((-100 : 100) ,(-100 : 100));
% xlabel('Target Output');
% ylabel('Model Output');
% zlabel('Output');
% saveas(h,'Scatter Test .fig');
    
    %finding weights and seperating it
    weightBias = formwb(net,net.b,net.iw,net.lw);
    [b,iw,lw] = separatewb(net,weightBias);
    
   
    %calculating net activation function of hidden layer 1 first node
    iw{1,1}(1,:)
    
    figure;
    xMin = min(testData(:,1));
    xMax = max(testData(:,1));
    
    yMin = min(testData(:,2));
    yMax = max(testData(:,2));

%      figure;                     %plotting output when training data is given
%     xMin = min(trainData(:,1));
%     xMax = max(trainData(:,1));
%     
%     yMin = min(trainData(:,2));
%     yMax = max(trainData(:,2));


%     beta = 0.2;
%      syms f(x,y)
%     f = tansig(beta*x*iw{1,1}(1,1) + beta*y*iw{1,1}(1,2)+beta*b{1,1}(1));
%     ezsurf(f,[xMin,xMax,yMin,yMax],[0,1,0]);
%     title('hidden layer 1 node 1');
%     colormap winter
%     
%     
%     
%     
%     
%      calculating net activation function of hidden layer 2nd node
%     iw{1,1}(1,:)
%     
%     
%      syms f(x,y)
%     f = tansig(beta*x*iw{1,1}(2,1) + beta*y*iw{1,1}(2,2)+beta*b{1,1}(2));
%     ezsurf(f,[xMin,xMax,yMin,yMax],[0,1,0]);
%     title('hidden layer 1 node 2');
%     colormap winter
%     
%     
%      calculating net activation function of hidden layer 3rd node
%     
%      syms f(x,y)
%     f = tansig(beta*x*iw{1,1}(3,1) + beta*y*iw{1,1}(3,2)+beta*b{1,1}(3));
%     ezsurf(f,[xMin,xMax,yMin,yMax],[0,1,0]);
%      title('hidden layer 1 node 3');
%     colormap winter
    
    
    
    %calculating net activation for node in second 
    
%     figure;
%     syms f(x,y)
%     f = [];
%     for i = 1 : 8
%         f = f + lw{2,1}(1,i)*tansig(beta*x*iw{1,1}(i,1) + beta*y*iw{1,1}(i,2)+beta*b{1,1}(i)) ;
%     end
%     f = f + b{2,1}(1,1);
%     f = tansig(f);
%     ezsurf(f,[xMin,xMax,yMin,yMax],[0,1,0]);
%     colormap winter

% storing the output of intermediate layers in f for 2st layer and g for
% 2nd layer

%%%% plotting output of first hidden node first hidden layer
% Z = cell(1,hidden1);
% totOut = [];
% for i = 1 : hidden1
% [p1,p2] = meshgrid(xMin:0.5:xMax);
% %func = tansig for hidden layers
% %func = purelin for output layer
% func = 'tansig';
% 
% 
% z = feval(func, [p1(:) p2(:)]*[iw{1,1}(i,1) iw{1,1}(i,2)]'+b{1,1}(i) );
% totOut = [totOut z];
% z = reshape(z,length(p1),length(p2));
% Z{1,i} = z;
% 
% grid on
% h = figure
% surf(p1,p2,z);
% titleName = ['output of first hidden layer node ' num2str(i) ' for epoch ' num2str(net.trainParam.epochs)];
% title(titleName);
% xlabel('Input 1');
% ylabel('Input 2');
% zlabel('Neuron output');
% colormap autumn
% saveas(h,[titleName '.fig']);
% 
% end
% 
% Z1 = cell(1,hidden2);
% 
% totInput = length(p1)*length(p2);
% output = totOut*net.lw{2}';
% totOut2 = [];
% %% ploting for second hidden layer
% for j = 1 : hidden2
%      
%     
%     act = output(:,j);
% z1 = feval(func, (act+b{2,1}(j)) );
% totOut2 = [totOut2 z1];
% z2 = reshape(z1,length(p1),length(p2));
% Z1{1,i} = z2;
% 
% grid on
% h = figure
% surf(p1,p2,z2);
% titleName = ['output of second hidden layer node ' num2str(j) ' for epoch ' num2str(net.trainParam.epochs)];
% 
% colormap autumn
% xlabel('Input 1');
% ylabel('Input 2');
% zlabel('Neuron output');
% saveas(h,[titleName '.fig']);
% end
% 
% func = 'purelin';
% output = totOut2*lw{3,2}';
% output = feval( func,output + b{3,1});
% 
% z3 = reshape(output,length(p1),length(p2));
% grid on
% h = figure
% surf(p1,p2,z3);
% titleName = ['output of output node for epoch ' num2str(net.trainParam.epochs)];
% 
% title('output of output node');
% colormap autumn
% %ezsurf(z,[xMin,xMax,yMin,yMax]);
% 
% xlabel('Input 1');
% ylabel('Input 2');
% zlabel('Neuron output');
% saveas(h,[titleName '.fig']);
%%%% my exp finish here

% beta = 0.1;
% A = sym('a',[1,hidden1]); % for first layer outputs
% for k = 1:numel(A)
%     A(k) = sym(sprintf('a%d', k));
% end
% 
% syms f(x,y)
% 
% for i = 1 : hidden1;
%  %syms f(x,y)
%     A(i) =  tansig(beta*x*iw{1,1}(i,1) + beta*y*iw{1,1}(i,2)+beta*b{1,1}(i));
%      figure;
%      ezsurf(A(i),[xMin,xMax,yMin,yMax],[0,1,0]);
%      title(['output of first hidden layer node' num2str(i)]);
%       colormap winter
% end
%     
% B = sym('b',[1,hidden2+1]); % for first layer outputs
% for k = 1:numel(B)
%     B(k) = sym(sprintf('b%d', k));
% end
% 
% for hid2 = 1 : hidden2
%     B(hid2) = 0;
%     for hid1 = 1 : hidden1
%         B(hid2) = B(hid2) +  lw{2,1}(hid2,hid1)*A(hid1);
%     end
%     B(hid2) = B(hid2) + b{2,1}(hid2);
%      B(hid2) =  tansig(beta * B(hid2));
%       figure;
%     ezsurf(B(hid2),[xMin,xMax,yMin,yMax],[0,1,0]);
%     title(['output of hidden layer 2 node' num2str(hid2)]);
%      colormap winter
% end
% 
% G = sym('g',[1,1]);
% for k = 1:numel(G)
%     G(k) = sym(sprintf('g%d', k));
% end
% for hid2 = 1 :4
%     G(1) = G(1) + B(hid2);
% end
% B(hidden2+1) = 0;
% for i = 1 : hidden2
%     B(hidden2+1) =  B(hidden2+1) + B(i)*lw{3,2}(i);
%     
% end
% B(hidden2+1) = B(hidden2+1) + b{3,1};
% %to plot the output of the OUTPUT LAYER
%  figure;
%     %ezsurf(G(1),[xMin,xMax,yMin,yMax]);
%   % ezsurf( B(1)*lw{3,2}(1) + B(2)*lw{3,2}(2)+ B(3)*lw{3,2}(3)+B(4)*lw{3,2}(4)+b{3,1},[xMin,xMax,yMin,yMax]);
%   ezsurf(B(hidden2+1),[xMin,xMax,yMin,yMax],[0,1,0]);
%    title('output of output node');
%      colormap winter
     
%      gridX = xMin : 0.2 : xMax;
%      gridY = yMin : 0.2: yMax;
%      [Xgrid,Ygrid] = meshgrid(gridX,gridY);
%      XgridNew = reshape(Xgrid,[size(Xgrid,1)*size(Xgrid,2),1]);
%      YgridNew = reshape(Ygrid,[size(Ygrid,1)*size(Ygrid,2),1]);
%      testDataGrid = [XgridNew YgridNew];
%      modelGrid = net(testDataGrid');
%      modelGridNew = reshape(modelGrid,[size(Xgrid,1),size(Ygrid,1)]);
%      h = figure;
%      surf(Xgrid,Ygrid,modelGridNew);
%      
%      
%      titleName = ['output of output node for epoch (Approx surface) ' num2str(net.trainParam.epochs)];
% 
% title('output of output node');
% colormap autumn
% %ezsurf(z,[xMin,xMax,yMin,yMax]);
% 
% xlabel('Input 1');
% ylabel('Input 2');
% zlabel('Neuron output');
% saveas(h,[titleName '.fig']);
end
figure
    plot(ep,performanceArr,'-r');
INPUT = load('input.mat');
TARGET = load('target.mat'); % for bivariate data




%%splitting data into train validation and target
trainData = INPUT.trainData;
trainDataTarget = INPUT.trainDataTarget;
trainDataFull = [trainData trainDataTarget];

valData = INPUT.valData;
valDataTarget = INPUT.valDataTarget;
valDataFull = [valData valDataTarget];


testData = INPUT.testData;
testDataTarget = INPUT.testDataTarget;
testDataFull = [testData testDataTarget];

%this section is for univivariate data
% INPUT = load('Train_101');
%  trainData = (INPUT.Input)';
% trainDataTarget = (INPUT.Target)';
%  trainDataFull = [trainData trainDataTarget];
%  
%  valData = (INPUT.valInput)';
% valDataTarget = (INPUT.valTarget)';
%  valDataFull = [valData valDataTarget];
% % 
% % 
%  testData = (INPUT.testInput)';
%  testDataTarget = (INPUT.testTarget)';
% testDataFull = [testData testDataTarget];


MC = 3:1 : 40;   %MC - model complexities 40 for bivariate 20 for univariate
logLambda = -30:1:0;  %regularization paramter
mSize = size(MC,2);
lSize = size(logLambda,2);
errorVal = zeros(mSize,lSize);
errorTrain = zeros(mSize,lSize);
errorTest = zeros(mSize,lSize);
weigths = cell(mSize,lSize);
centroids = cell(mSize,1);
Distances = cell(mSize,1);
alphaArr = zeros(mSize,1);
for m = 1 : mSize
    [idxT,CTrain,sumd,DTrain] = kmeans(trainData,MC(m));
    alphaTrain = computeAlpha(CTrain);
    widthTrain = alphaTrain/sqrt(2*MC(m));
    phiTrain = findPHI(DTrain,widthTrain,CTrain,trainData);
    centroids{m,1} = CTrain;
    Distances{m,1} = DTrain;
    alphaArr(m,1)= alphaTrain;
%     [idxV,CVal,sumd,DVal] = kmeans(valDataFull,MC(m));
%     alphaVal = computeAlpha(CVal);
%     widthVal = alphaVal/sqrt(2*MC(m));
%     phiVal = findPHI(DVal,widthVal);
    
%      [idxTe,CTest,sumd,DTest] = kmeans(testDataFull,MC(m));
%     alphaTest = computeAlpha(CTest);
%     widthTest = alphaTest/sqrt(2*MC(m));
%     phiTest = findPHI(DTest,widthTest);
    
    
    phiTilda = findPHITILDA(CTrain,widthTrain);
    for l = 1: lSize
        weights{m,l} = findWeights(phiTrain,phiTilda,trainDataTarget,logLambda(l));
        errorVal(m,l) = computeError(weights{m,l},valData,valDataTarget,CTrain,widthTrain,logLambda(l),phiTilda);
        errorTrain(m,l) = computeError(weights{m,l},trainData,trainDataTarget,CTrain,widthTrain,logLambda(l),phiTilda);
        errorTest(m,l) = computeError(weights{m,l},testData,testDataTarget,CTrain,widthTrain,logLambda(l),phiTilda);
    end
    
end

minE = min(min(errorVal));
[r,c] = find(minE == errorVal,1);
%%for trial and error manually giving r and c

%% for seeing output for lamba = -18
% c1 = find(logLambda == -18);
%optWeights = weights{r,c};

optWeights = weights{r,c};

optCentroids = centroids{r,1};
%computing the optimal centroids
alphanew = alphaArr(r,1);
widthnew = alphanew/sqrt(2*MC(r));



modelTrain = findModel(optWeights, trainData,optCentroids,widthnew);
modelVal = findModel(optWeights, valData,optCentroids,widthnew);
modelTest = findModel(optWeights, testData,optCentroids,widthnew);


 h =  figure;
scatter3(valDataFull(:,1),valDataFull(:,2),valDataFull(:,3),15,[0 0 1],'filled');
hold on;
scatter3(valDataFull(:,1),valDataFull(:,2),modelVal,30,[1 0 0],'filled');
title('Model output vs Target output for Validation data ');

legend('Target Output','Model Output');
xlabel('Input variable x1');
ylabel('Input variable x2');
zlabel('Output');
axis([-10 10 -10 10 -100 100]);
view([-35,35]);
saveas(h,'TOMO Val GRBFNN_BIVariate.fig');
% 
h =  figure;
scatter3(testDataFull(:,1),testDataFull(:,2),testDataFull(:,3),15,[0 0 1],'filled');
hold on;
scatter3(testDataFull(:,1),testDataFull(:,2),modelTest,30,[1 0 0],'filled');
title('Model output vs Target output for Test data ');

legend('Target Output','Model Output');
xlabel('Input variable x1');
ylabel('Input variable x2');
zlabel('Output');
axis([-10 10 -10 10 -100 100]);
view([-35,35]);
saveas(h,'TOMO Test GRBFNN_BIVariate.fig');
% 
 h = figure;
scatter(testDataFull(:,3),modelTest);
hold on;
title('scatter plot of test data');
plot(-100:100,-100:100,'-r');
xlabel('Target');
ylabel('Model');
saveas(h,'Scatter Test GRBFNN_BIVariate.png');


h = figure;
scatter(valDataFull(:,3),modelVal);
hold on;
title('scatter plot of val data');
plot(-100:100,-100:100,'-r');
xlabel('Target');
ylabel('Model');
saveas(h,'Scatter Validation GRBFNN_BIVariate.png');


h = figure;
scatter(trainDataFull(:,3),modelTrain);
hold on;
title('scatter plot of train data');
plot(-100:100,-100:100,'-r');
xlabel('Target');
ylabel('Model');
saveas(h,'Scatter Train GRBFNN_BIVariate.png');
% 
% h = figure;
% scatter(trainDataFull(:,3),modelTrain,15,'filled');
% hold on;
% title('scatter plot of train data');
% plot(-100:100,-100:100,'-r');
% 
h =  figure;
scatter3(trainDataFull(:,1),trainDataFull(:,2),trainDataFull(:,3),15,[0 0 1],'filled');
hold on;
scatter3(trainDataFull(:,1),trainDataFull(:,2),modelTrain,30,[1 0 0],'filled');
title('Model output vs Target output for Train data ');

legend('Target Output','Model Output');
xlabel('Input variable x1');
ylabel('Input variable x2');
zlabel('Output');
axis([-10 10 -10 10 -100 100]);
view([-35,35]);
saveas(h,'TOMO Train GRBFNN_BIVariate.fig');


%%to plot the surface of the test points
 
    xMin = min(trainDataFull(:,1));
    xMax = max(trainDataFull(:,1));
    
    yMin = min(trainDataFull(:,2));
    yMax = max(trainDataFull(:,2));
    
     zMin = min(valDataFull(:,3));
    zMax = max(valDataFull(:,3));
    
%%following code is to plot the surface of the model
     gridX = xMin : 0.5 : xMax;
     gridY = yMin : 0.5: yMax;
      
      zSep = (zMax - zMin)/(size(gridY,2) - 1);
 gridZ = zMin : zSep : zMax;
     
     
     [Xgrid,Ygrid] = meshgrid(gridX,gridY);
     XgridNew = reshape(Xgrid,[size(Xgrid,1)*size(Xgrid,2),1]);
     YgridNew = reshape(Ygrid,[size(Ygrid,1)*size(Ygrid,2),1]);
     testDataGrid = [XgridNew YgridNew];
     modelGrid = findModel(optWeights, testDataGrid,optCentroids,widthnew);
     modelGridNew = reshape(modelGrid,[size(Xgrid,1),size(Ygrid,1)]);
     h = figure;
     surf(Xgrid,Ygrid,modelGridNew);
     colormap(autumn)
     hold on
     scatter3(testDataFull(:,1),testDataFull(:,2),testDataFull(:,3),30,[0 0 1],'filled');
     legend('Approximated Surface','Test Data','Location','Best');
     saveas(h,'Approx Surf GRB with Test.fig');
 %%following code is to plot surface of the given model
 
 h = figure;
     surf(Xgrid,Ygrid,modelGridNew);
     colormap(autumn)
     hold on
     scatter3(trainDataFull(:,1),trainDataFull(:,2),trainDataFull(:,3),30,[0 0 1],'filled');
     legend('Approximated Surface','Train Data','Location','Best');
     saveas(h,'Approx Surf GRB with Train.fig');
     
      h = figure;
     surf(Xgrid,Ygrid,modelGridNew);
     colormap(autumn)
     hold on
     scatter3(valDataFull(:,1),valDataFull(:,2),valDataFull(:,3),30,[0 0 1],'filled');
     legend('Approximated Surface','Val Data','Location','Best');
     saveas(h,'Approx Surf GRB with Val.fig');
     
     
     

 
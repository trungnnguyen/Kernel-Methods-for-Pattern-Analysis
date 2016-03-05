INPUT = load('Train_1001.mat');
 trainData = (INPUT.Input)';
trainDataTarget = (INPUT.Target)';
 trainDataFull = [trainData trainDataTarget];
 
 valData = (INPUT.valInput)';
valDataTarget = (INPUT.valTarget)';
 valDataFull = [valData valDataTarget];
% 
% 
 testData = (INPUT.testInput)';
 testDataTarget = (INPUT.testTarget)';
testDataFull = [testData testDataTarget];
%10 is for training size 15
MC = 3:1 : 20;   %%MC - model complexities 40 for bivariate 20 for univariate and size of training data
logLambda = -25:1:0;  %regularization paramter
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

    
    
    phiTilda = findPHITILDA(CTrain,widthTrain);
    for l = 1 : lSize
        weights{m,l} = findWeights(phiTrain,phiTilda,trainDataTarget,logLambda(l));
        errorVal(m,l) = computeError(weights{m,l},valData,valDataTarget,CTrain,widthTrain,logLambda(l),phiTilda);
        errorTrain(m,l) = computeError(weights{m,l},trainData,trainDataTarget,CTrain,widthTrain,logLambda(l),phiTilda);
        errorTest(m,l) = computeError(weights{m,l},testData,testDataTarget,CTrain,widthTrain,logLambda(l),phiTilda);
    end
    
end

minE = min(min(errorVal));
[r,c] = find(minE == errorVal,1);

%%Initialising c

optWeights = weights{r,c};

optCentroids = centroids{r,1};
%computing the optimal centroids
alphanew = alphaArr(r,1);
widthnew = alphanew/sqrt(2*MC(r));



% modelTrain = findModel(optWeights, trainData,optCentroids,widthnew);
% modelVal = findModel(optWeights, valData,optCentroids,widthnew);
% modelTest = findModel(optWeights, testData,optCentroids,widthnew);
DATA = (INPUT.X)';
model = findModel(optWeights, DATA,optCentroids,widthnew);
modelTrain = model(INPUT.TrainInd);
modelTest = model(INPUT.TestIndices);
modelVal = model(INPUT.ValIndices);

set(gca,'fontsize',16);
h = figure
set(gca,'fontsize',16);

plot(trainData,modelTrain,'or',trainData,trainDataTarget,'ob','LineWidth',2);
title(['Target Vs Model for cluster size' num2str(MC(r)) ' ln\lambda = ' num2str(logLambda(c))]);
xlabel('Input');
ylabel('Output');
hold on;
plot(INPUT.X,INPUT.Y,'-g','LineWidth',1);
legend('Model Output','Target Output','Location','Best');
saveas(h,['TOMO Train GRBF_UNI_TrainSize' num2str(size(trainData,1)) '.png']);
close(h);

h = figure
set(gca,'fontsize',16);
plot(valData,modelVal,'or',valData,valDataTarget,'ob','LineWidth',2);
title(['Target Vs Output Data for clusters ' num2str(MC(r)) ' ln\lambda = ' num2str(logLambda(c))]);
xlabel('Input');
ylabel('Output');
hold on;
plot(INPUT.X,INPUT.Y,'-g','LineWidth',1);
legend('Model Output','Target Output','Location','Best');
saveas(h,['TOMO Val GRBF_UNI_TrainSize' num2str(size(trainData,1)) '.png']);

legend('Model Output','Target Output');
close(h);

h = figure
set(gca,'fontsize',16);
plot(testData,modelTest,'or',testData,testDataTarget,'ob','LineWidth',2);
title(['Target Vs Output for Test Data for clusters ' num2str(MC(r)) ' ln\lambda = ' num2str(logLambda(c))]);
xlabel('Input');
ylabel('Output');
hold on;
plot(INPUT.X,INPUT.Y,'-g','LineWidth',1);

legend('Model Output','Target Output','Location','Best');

saveas(h,['TOMO Test GRBF_UNI_TrainSize' num2str(size(trainData,1)) '.png']);
close(h);


%%plot for approximated function
h = figure
set(gca,'fontsize',16);
plot(DATA',model,'-r','LineWidth',3);
hold on
plot(INPUT.X,INPUT.Y,'-g','LineWidth',2);
hold on
plot(trainData,trainDataTarget,'ob','LineWidth',0.3);
legend('Approximated Function','Function','Training Data','Location','Best');
xlabel('Input');
ylabel('Output');
saveas(h,['Approx Func GRBF_UNI_TrainSize' num2str(size(trainData,1)) '.png']);

close(h);
%%scatter plot for the entire data
h = figure
set(gca,'fontsize',16);
scatter(INPUT.Y_n,model,'LineWidth',2);
hold on;
plot(0:4 ,0:4,'-r','LineWidth',2);
xlabel('Target');
ylabel('Model');
title('Scatter Plot for entire data');
saveas(h,['Scatter Entire Data GRBF_UNI_TrainSize' num2str(size(trainData,1)) '.png']);
close (h)

%%scatter plot of training data
h = figure
set(gca,'fontsize',16);
scatter(trainDataTarget,modelTrain,'LineWidth',2);
hold on;
plot(0:4 ,0:4,'-r','LineWidth',2);
xlabel('Target Output');
ylabel('Model Output');
title('Scatter Plot for Training data');
saveas(h,['Scatter Train Data GRBF_UNI_TrainSize' num2str(size(trainData,1)) '.png']);



%%scatter plot of Val data
h = figure
set(gca,'fontsize',16);
scatter(valDataTarget,modelVal,'LineWidth',2);
hold on;
plot(0:4 ,0:4,'-r','LineWidth',2);
xlabel('Target Output');
ylabel('Model Output');
title('Scatter Plot for Validation data');
saveas(h,['Scatter Val Data GRBF_UNI_TrainSize' num2str(size(trainData,1)) '.png']);

%%scatter plot of Test data
h = figure
set(gca,'fontsize',16);
scatter(testDataTarget,modelTest,'LineWidth',2);
hold on;
plot(0:4 ,0:4,'-r','LineWidth',2);
xlabel('Target Output');
ylabel('Model Output');


title('Scatter Plot for Test data');
saveas(h,['Scatter Test Data GRBF_UNI_TrainSize' num2str(size(trainData,1)) '.png']);

h = figure
set(gca,'fontsize',16);
plot(logLambda,errorVal(r,:),'-r','LineWidth',2);
hold on
plot(logLambda,errorTrain(r,:),'-g','LineWidth',2);
plot(logLambda,errorTest(r,:),'-b','LineWidth',2);
xlabel('ln\lambda')
ylabel('E_{RMS}');
legend('Val Error','Train Error','Test Error','Location','Best');


h = figure
set(gca,'fontsize',16);
plot(MC,errorVal(:,c),'-r','LineWidth',2);
hold on
plot(MC,errorTrain(:,c),'-g','LineWidth',2);
plot(MC,errorTest(:,c),'-b','LineWidth',2);
xlabel('Model Complexity(Number Of Clusters)')
ylabel('E_{RMS}');
legend('Val Error','Train Error','Test Error','Location','Best');







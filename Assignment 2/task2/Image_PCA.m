%featureMat = load('feature.txt');
load('Group_6_data.mat');

%%loading Data into respective Classes


class1Data = [];
class2Data = [];
class3Data = [];
class4Data = [];
class5Data = [];
Labels = [];
feature_output = load('feature_output.mat');
rows = size(feature_output,1);
for i  = 1 : 
   % info = featureMat(i,:);
   % label = feature_output(i,2:6);
    index = find(label == 1);
    Labels = [Labels;index];
    if index == 1
%         class1Data = [class1Data;info];
%         dlmwrite('class1Data.txt',class1Data,' ');
count1 = count2 + 1;
    end
    if index == 2
%         class2Data = [class2Data;info];
%         dlmwrite('class2Data.txt',class2Data,' ');
count2 = count2 + 1;
    end
    if index == 3
%         class3Data = [class3Data;info];
%         dlmwrite('class3Data.txt',class3Data,' ');
count3 = count3 + 1;
    end
    if index == 4
%         class4Data = [class4Data;info];
%         dlmwrite('class4Data.txt',class4Data,' ');
count4 = count4 + 1;
    end
    if index == 5
%         class5Data = [class5Data;info];
%         dlmwrite('class5Data.txt',class5Data,' ');
count5 = count5 + 1;
    end
    
    
end

trainPercent = 0.7;
valPercent = 0.2;
testPercent = 0.1;
%dlmwrite('feature.txt',featureOut,' ');

% Size = size(class5Data,1);
% trainSize = floor(trainPercent * Size);
% valSize = floor(valPercent * Size);
% 
% testSize = ceil(testPercent * Size);
% 
% class5Train = class5Data(1 : trainSize,:);
% class5Val = class5Data(trainSize + 1 : valSize + trainSize,:);
% class5Test = class5Data(valSize + trainSize+1: Size,:);


class1Train = load('Task2/class1Train.mat');
class1Train = class1Train.class1Train;

class2Train = load('Task2/class2Train.mat');
class2Train = class2Train.class2Train;

class3Train = load('Task2/class3Train.mat');
class3Train = class3Train.class3Train;

class4Train = load('Task2/class4Train.mat');
class4Train = class4Train.class4Train;

class5Train = load('Task2/class5Train.mat');
class5Train = class5Train.class5Train;



class1Val = load('Task2/class1Val.mat');
class1Val = class1Val.class1Val;

class2Val = load('Task2/class2Val.mat');
class2Val = class2Val.class2Val;

class3Val = load('Task2/class3Val.mat');
class3Val = class3Val.class3Val;

class4Val = load('Task2/class4Val.mat');
class4Val = class4Val.class4Val;


class5Val = load('Task2/class5Val.mat');
class5Val = class5Val.class5Val;

class1Test = load('Task2/class1Test.mat');
class1Test = class1Test.class1Test;
class2Test = load('Task2/class2Test.mat');
class2Test = class2Test.class2Test;

class3Test = load('Task2/class3Test.mat');
class3Test = class3Test.class3Test;
class4Test = load('Task2/class4Test.mat');
class4Test = class4Test.class4Test;
class5Test = load('Task2/class5Test.mat');
class5Test = class5Test.class5Test;

TrainData = [class1Train;class2Train;class3Train;class4Train;class5Train];
ValData = [class1Val;class2Val;class3Val;class4Val;class5Val];
TestData = [class1Test;class2Test;class3Test;class4Test;class5Test];

TrainLabels = [ones(size(class1Train,1),1);2*ones(size(class2Train,1),1);3*ones(size(class3Train,1),1);4*ones(size(class4Train,1),1);5*ones(size(class5Train,1),1)];
TestLabels = [ones(size(class1Test,1),1);2*ones(size(class2Test,1),1);3*ones(size(class3Test,1),1);4*ones(size(class4Test,1),1);5*ones(size(class5Test,1),1)];
ValLabels = [ones(size(class1Val,1),1);2*ones(size(class2Val,1),1);3*ones(size(class3Val,1),1);4*ones(size(class4Val,1),1);5*ones(size(class5Val,1),1)];

%%deciding  kernel by seeing the kernel gram matrix%%%


X = dist(TrainData,TrainData');
minDist = min(min(X));
maxDist = max(max(X));

X = X.^2; %% calculating the sqaure of distances
gammaArray = [0.001,0.003,0.005,0.01,0.02,0.05,0.1];
C  = [size(class1Train,1),size(class2Train,1),size(class3Train,1),size(class4Train,1),size(class5Train,1)];
errArray = [];
minError = Inf;
Gmin = zeros(size(X));
minGama = -1;
for i = 1 :size(gammaArray,2)
[G,err]=computeKernelGram(X,gammaArray(i),C);
errArray = [errArray err];
if err < minError
    minGama = gammaArray(i);
    minError = err;
    Gmin = G;
end
end
imshow(Gmin);
minGama



arg = [];
maxAccuracy = 0;
bestArg = [];
bestNu = 0;
bestModel = [];
accuracyValArr = [];
accuracyTrainArr = [];
accuracyTestArr = [];
nSVArray = [];
for nuP = 0.01 : 0.02:0.4
    arg = [];
    arg = [arg,'-s 1 -t 2 -g 0.1 -b 1 -n ']
    arg =  [arg,num2str(nuP),' -q'];
model = svmtrain(TrainLabels,TrainData,arg);

% [predicted_label, accuracy, decision_values] = svmpredict(TrainLabels, TrainData, model, '-b 1');%, '-s 1 -t 2 -g 0.model = svmtrain(TrainLabels,TrainData,'-s 1 -t 2 -g 0.2 -n 0.05 -q');
% accuracyTrainArr = [accuracyTrainArr accuracy(1,1)]

[predicted_label, accuracy, decision_values] = svmpredict(ValLabels, ValData, model, '-b 1');%, '-s 1 -t 2 -g 0.model = svmtrain(TrainLabels,TrainData,'-s 1 -t 2 -g 0.2 -n 0.05 -q');
accuracyValArr = [accuracyValArr accuracy(1,1)]

% [predicted_label, accuracy, decision_values] = svmpredict(TestLabels, TestData, model, '-b 1');%, '-s 1 -t 2 -g 0.model = svmtrain(TrainLabels,TrainData,'-s 1 -t 2 -g 0.2 -n 0.05 -q');
% accuracyTestArr = [accuracyTestArr accuracy(1,1)]


nSVArray = [nSVArray model.totalSV];
if maxAccuracy < accuracy(1,1)
    maxAccuracy = accuracy(1,1);
    bestArg = arg;
    bestNu = nuP;
    bestModel = model;
end
end

maxAccuracy
bestArg
bestNu
bestModel
nuPArr =  0.01 : 0.02:0.4;

% figure;
% h = plot(nuPArr,100 - accuracyValArr,'-r','LineWidth',2);
% 
% %title('nu Values Vs Accuracy');
% %saveas(h,'nuVsAccuValWithoutPCAgama10.eps');
% hold on;
% 
% plot(nuPArr,100 - accuracyTrainArr,'-b','LineWidth',2);
% 
% %title('nu Values Vs Accuracy');
% %saveas(h,'nuVsAccuTrainWithoutPCAgama10.eps');
% hold on;
% 
%  plot(nuPArr,100 - accuracyTestArr,'-g','LineWidth',2);
% xlabel('nu');
% ylabel('error');
% title('nu Values Vs Error');
% 
% legend('Validation Error','Train Error','Test Error','Location','Best')
% saveas(h,'nuVsAccuWithoutPCAgama10.png');

h = plot(nuPArr,nSVArray,'-b','LineWidth',2);
xlabel('nu');
ylabel('number OF support vectors');
title('nu Values Vs SupportVectors');
saveas(h,'nuVsSVWithoutPCAgama10.eps');

[predicted_label, accuracy, decision_values] = svmpredict(TestLabels, TestData, bestModel);
r = confusionmat(TestLabels,predicted_label);
h = plotconfusion(ind2vec(TestLabels'),ind2vec(predicted_label'));
saveas(h,'confusionwithoutPCAgama10.eps');
acc = (sum(diag(r)))/(sum(sum(r)));


%%Lets do
%%PCA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inputPCA = TrainData;
COV = cov(inputPCA);

[eVect,eVal] = eig(COV);

eVal = diag(eVal);
[sortedEVal,Ind] = sort(eVal,'descend');
hold on
h = plot(Ind,sortedEVal,'*');
xlabel('Indices');
ylabel('Eigen Values');
saveas(h,'Plot of Eig.fig');
saveas(h,'Plot of Eig.eps');
hold on
%%taking top 58 eigen values take with first 25 and 58 0.001 for 58th value

eVectNew = eVect(:,Ind);

eVectNew = eVectNew(:,1:10);

newTrain = TrainData*(eVectNew);
newTest  = TestData*(eVectNew);
newVal = ValData*(eVectNew);

TrainData = newTrain;
TestData = newTest;
ValData = newVal;

X = dist(newTrain,newTrain');
minDist = min(min(X));
maxDist = max(max(X));

X = X.^2; %% calculating the sqaure of distances
gammaArray = [0.001,0.003,0.005,0.01,0.02,0.05,0.1];
C  = [size(class1Train,1),size(class2Train,1),size(class3Train,1),size(class4Train,1),size(class5Train,1)];
errArray = [];
minError = Inf;
Gmin = zeros(size(X));
minGama = -1;
for i = 1 :size(gammaArray,2)
[G,err]=computeKernelGram(X,gammaArray(i),C);
errArray = [errArray err];
if err < minError
    minGama = gammaArray(i);
    minError = err;
    Gmin = G;
end
end
imshow(Gmin);
minGama;

% %% model slection
arg = [];
maxAccuracy = 0;
bestArg = [];
bestNu = 0;
bestModel = [];
accuracyValArr = [];
accuracyTrainArr = [];
accuracyTestArr = [];
nSVArray = [];
for nuP = 0.01 : 0.02:0.4
    arg = [];
    arg = [arg,'-s 1 -t 2 -g 0.1 -b 1 -n ']
    arg =  [arg,num2str(nuP),' -q'];
model = svmtrain(TrainLabels,TrainData,arg);

[predicted_label, accuracy, decision_values] = svmpredict(TrainLabels, TrainData, model, '-b 1');%, '-s 1 -t 2 -g 0.model = svmtrain(TrainLabels,TrainData,'-s 1 -t 2 -g 0.2 -n 0.05 -q');
accuracyTrainArr = [accuracyTrainArr accuracy(1,1)]

[predicted_label, accuracy, decision_values] = svmpredict(ValLabels, ValData, model, '-b 1');%, '-s 1 -t 2 -g 0.model = svmtrain(TrainLabels,TrainData,'-s 1 -t 2 -g 0.2 -n 0.05 -q');
accuracyValArr = [accuracyValArr accuracy(1,1)]

[predicted_label, accuracy, decision_values] = svmpredict(TestLabels, TestData, model, '-b 1');%, '-s 1 -t 2 -g 0.model = svmtrain(TrainLabels,TrainData,'-s 1 -t 2 -g 0.2 -n 0.05 -q');
accuracyTestArr = [accuracyTestArr accuracy(1,1)]


nSVArray = [nSVArray model.totalSV];
if maxAccuracy < accuracy(1,1)
    maxAccuracy = accuracy(1,1);
    bestArg = arg;
    bestNu = nuP;
    bestModel = model;
end
end

maxAccuracy
bestArg
bestNu
bestModel
nuPArr =  0.01 : 0.02:0.4;

figure;
h = plot(nuPArr,100 - accuracyValArr,'-r','LineWidth',2);

%title('nu Values Vs Accuracy');
%saveas(h,'nuVsAccuValWithoutPCAgama10.eps');
hold on;

plot(nuPArr,100 - accuracyTrainArr,'-b','LineWidth',2);

%title('nu Values Vs Accuracy');
%saveas(h,'nuVsAccuTrainWithoutPCAgama10.eps');
hold on;

 plot(nuPArr,100 - accuracyTestArr,'-g','LineWidth',2);
xlabel('nu');
ylabel('error');
title('nu Values Vs Error');

legend('Validation Error','Train Error','Test Error','Location','Best')
saveas(h,'nuVsAccuWithPCAgama10top10.png');

h = plot(nuPArr,nSVArray,'-b','LineWidth',2);
xlabel('nu');
ylabel('number OF support vectors');
title('nu Values Vs SupportVectors');
saveas(h,'nuVsSVWithPCAgama10top10.eps');

[predicted_label, accuracy, decision_values] = svmpredict(TestLabels, newTest, bestModel);
r = confusionmat(TestLabels,predicted_label);
plotconfusion(ind2vec(TestLabels'),ind2vec(predicted_label'));

acc = (sum(diag(r)))/(sum(sum(r)));





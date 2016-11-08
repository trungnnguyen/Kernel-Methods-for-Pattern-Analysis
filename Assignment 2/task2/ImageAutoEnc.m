
clear
clc
TrainLabels = load('TrainLabels.mat');
TrainLabels = TrainLabels.TrainLabels;

TestLabels = load('TestLabels.mat');

TestLabels = TestLabels.TestLabels;

ValLabels  = load('ValLabels.mat');
ValLabels = ValLabels.ValLabels;

TrainData = load('DataStackAutoEnc/stTrain100.mat');
TrainData = TrainData.x;


TestData = load('DataStackAutoEnc/stTest100.mat');
TestData = TestData.x;

ValData = load('DataStackAutoEnc/stVal100.mat');
ValData = ValData.x;

% 
% X = pdist2(TrainData,TrainData);
% 
% 
% X = X.^2; %% calculating the sqaure of distances
% gammaArray = [0.001,0.003,0.005,0.01,0.02,0.05,0.1];
% C = load('sizeTrain.mat');
% C =C.C;
% errArray = [];
% minError = Inf;
% Gmin = zeros(size(X));
% minGama = -1;
% for i = 1 :size(gammaArray,2)
% [G,err]=computeKernelGram(X,gammaArray(i),C);
% errArray = [errArray err];
% if err < minError
%     minGama = gammaArray(i);
%     minError = err;
%     Gmin = G;
% end
% end
% imshow(Gmin);
% minGama



arg = [];
maxAccuracy = 0;
bestArg = [];
bestNu = 0;
bestModel = [];
accuracyArr = [];
nSVArray = [];
for nuP = 0.01 : 0.02:0.4
    arg = [];
    arg = [arg,'-s 1 -t 2 -g 0.1 -b 1 -n ']
    arg =  [arg,num2str(nuP),' -q'];
model = svmtrain(TrainLabels,TrainData,arg);
[predicted_label, accuracy, decision_values] = svmpredict(ValLabels, ValData, model, '-b 1');%, '-s 1 -t 2 -g 0.model = svmtrain(TrainLabels,TrainData,'-s 1 -t 2 -g 0.2 -n 0.05 -q');
accuracyArr = [accuracyArr accuracy(1,1)]
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

figure
h = plot(nuPArr,accuracyArr,'-r','LineWidth',2);
xlabel('nu');
ylabel('accuracy');
title('nu Values Vs Accuracy');
%saveas(h,'nuVsAccuStackAutoEnc150gamaPoint1.eps');

figure
h = plot(nuPArr,nSVArray,'-b','LineWidth',2);
xlabel('nu');
ylabel('number OF support vectors');
title('nu Values Vs SupportVectors');
%saveas(h,'nuVsSVAutoStackAutoEncPoint1.eps');

[predicted_label, accuracy, decision_values] = svmpredict(TestLabels, TestData, bestModel);
r = confusionmat(TestLabels,predicted_label);
h = plotconfusion(ind2vec(TestLabels'),ind2vec(predicted_label'));
%saveas(h,'confusionStackAutoEncPoint1.eps');
acc = (sum(diag(r)))/(sum(sum(r)));

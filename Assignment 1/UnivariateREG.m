
%  X = 0:0.002:1;
%  
%  Y = exp(tanh(2*pi*X));
%  Y_n = Y + normrnd(0,0.3,1,size(Y,2));
%   figure;
%  plot(X,Y);
%  hold on
%  plot(X,Y_n,'or');
%  
%  Train = [X'  Y_n'];
%  %save Train_501
%  INPUT = load('Train_501');
%  X = INPUT.X;
%  Y = INPUT.Y;
% Y_n = INPUT.Y_n;
% X_limit = size(X,2);
% TrainInd = INPUT.TrainInd;
% TestIndices = INPUT.TestIndices;
% ValIndices = INPUT.ValIndices;
% % 
%  TrainSize = length(TrainInd);
%  ValSize = length(ValIndices);
% % % TrainSize = ceil(0.6 * X_limit);
% % % ValSize = floor(0.25 * X_limit);
% %  TrainInd = randperm( X_limit,TrainSize);
% % % 
% %  RemInd = setdiff(1:X_limit,TrainInd);
% %  ValInd = randperm(size(RemInd,2),ValSize);
% %  ValIndices = RemInd(ValInd);
% % % 
% %  TestIndices = setdiff(RemInd,ValIndices);
% % 
% % 
% Target = Y_n(TrainInd);
% Input = X(TrainInd);          %X y_new for 21 sample X_new and Y_new for larger samples
% 
% testTarget = Y_n(TestIndices);
% testInput = X(TestIndices);
% 
% valTarget = Y_n(ValIndices);
% valInput  = X(ValIndices);
% 
% figure;
% plot(testInput,testTarget,'ob');
% title('Test')
% 
% figure
% plot(valInput,valTarget,'ob');
% title('Val');
% 
% figure
% plot(Input,Target,'ob');
% title('Target');
% 
% M_Array = [0,1,3,5,8,11,14];
% M_ArrNew = [0,1,3,4,5,6,14]; %for larger samples
% M_interest = [0,1,3,4,5,6,14];
% TrainERMS = [];
% TestERMS = [];
% ValERMS = [];
% weightCell = cell(1,size(M_ArrNew,2));
% minError = Inf;
% minIndex = -1;
% for mIter =0 :14 %0 : 14 14 points is 15 weights for 15 points -------  19 for larger samples
% 
% M = mIter;
% 
% phi = zeros(TrainSize,M+1);
% for r = 1 : TrainSize
%     for c = 1 : M+1
%         phi(r,c) = Input(r)^(c-1);
%     end
% end
% if M == 14
%     phi14 = phi;
% end
% if M == 20
%     phi20 = phi;
% end
% W = pinv(phi)* Target'; %without regularization
% 
% 
% 
% ModelOutput = zeros(1,size(X,2)); %changed X to X_new
% for i = 1 : size(X,2)
%     ModelOutput(i) = 0;
%     for m = 1 : M+1
%         ModelOutput(i) = ModelOutput(i) + W(m)*(X(i)^(m-1));
%     end
% end
% 
% %%calulate TrainERMS
% ModelTrain = ModelOutput(TrainInd);
% TrainERMS(mIter+1) = sqrt(sum(((Target - ModelTrain).*(Target - ModelTrain))/size(Target,2)));
% 
% %%calculate TestERMS
% ModelTest = ModelOutput(TestIndices);
% TestERMS(mIter+1) = sqrt(sum(((testTarget - ModelTest).*(testTarget - ModelTest))/size(testTarget,2)));
% 
% ModelVal = ModelOutput(ValIndices);
% ValERMS(mIter + 1) = sqrt(sum(((valTarget - ModelVal).*(valTarget - ModelVal))/size(valTarget,2)));
% 
% if minError > ValERMS(mIter + 1)
%     minError = ValERMS(mIter + 1);
%     minIndex = mIter;
%     Mout = ModelOutput;
% end
% if ismember(mIter,M_interest) == 1   %M_ArrNew is for regularization M_Interest for scatter
% h = figure
% plot (X,Y,'LineWidth',2);
% weightCell{1,mIter+1} = W;
% hold on;
% scatter (Input,Target,'o');
% hold on;
% plot (X,ModelOutput,'-r','LineWidth',2);
% xlabel('Input');
% ylabel('Output');
% title(['I/p Vs O/p for M = ' num2str(M)]);
% axis([0 1 0 4]);
% legend('Function','Training Data','Approximated Function','Location','northoutside','Orientation','horizontal');
% saveas(h, ['for' num2str(X_limit) ' samples Ip Vs op '  num2str(M) '.fig']);
% 
% 
% 
% hold off;
% 
% h = figure;
% plot(Input,Target,'or',Input,ModelTrain,'ob');
% title(['Target Output Vs Model Output for Training data for M = ' num2str(M)]);
% hold on;
% xlabel('Input');
% ylabel('Output');
%        plot(X,Y,'-g','LineWidth',0.5);
%         legend('Target Output','Model Output','Fuction');
% 
%      saveas(h, ['for' num2str(X_limit) ' samplesTOMOTrain' num2str(M) '.fig']); 
%     close(h);
%     
%     h = figure;
% plot(testInput,testTarget,'or',testInput,ModelTest,'ob');
% hold on
% title(['Target Output Vs Model Output for Testing data for M = ' num2str(M)]);
% xlabel('Input');
% ylabel('Output');
% plot(X,Y,'-g','LineWidth',0.5);
%         legend('Target Output','Model Output','Fuction');
%      saveas(h, ['for' num2str(X_limit) ' samplesTOMOTest' num2str(M) '.fig']); 
%     close(h);
%     
%     h = figure;
% plot(valInput,valTarget,'or',valInput,ModelVal,'ob');
% title(['Target Output Vs Model Output for Validation data for M = ' num2str(M)]);
% xlabel('Input');
% ylabel('Output');
% hold on
% plot(X,Y,'-g','LineWidth',0.5);
%         legend('Target Output','Model Output','Fuction');
%      saveas(h, ['for' num2str(X_limit) ' samples TOMOValidation ' num2str(M) '.fig']); 
%     close(h);
%     
%     W
%     
%     h = figure;
%     scatter(Target,ModelTrain,'filled');
%     title(['scatter plot of Target vs Model Output for training data' num2str(M)]);
%     hold on;
%     plot(0:4 ,0:4,'-r')
%     xlabel('Target Output');
%     ylabel('Model Output');
%     saveas(h, ['for' num2str(X_limit) ' samples ScatterTrain ' num2str(M) '.fig']); 
%     close(h);
%     
%     h = figure;
%     scatter(valTarget,ModelVal,'filled');
%     title(['scatter plot of Target vs Model Output for Validation data' num2str(M)]);
%     hold on;
%     plot(0:4 ,0:4,'-r')
%     xlabel('Target Output');
%     ylabel('Model Output');
%     saveas(h, ['for' num2str(X_limit) ' samples ScatterVal ' num2str(M) '.fig']); 
%     close(h);
%     
%     h = figure;
%     scatter(testTarget,ModelTest,'filled');
%     title(['scatter plot of Target vs Model Output for Testing data' num2str(M)]);
%     hold on;
%     plot(0:4 ,0:4,'-r')
%     xlabel('Target Output');
%     ylabel('Model Output');
%     saveas(h, ['for' num2str(X_limit) ' ScatterTest'  num2str(M) '.fig']); 
%     close(h);
% 
% end
% end
% 
% h = figure
% plot (X,Y,'LineWidth',2);
% 
% hold on;
% scatter (Input,Target,'o');
% hold on;
% plot (X,Mout,'-r','LineWidth',2);
% xlabel('Input');
% ylabel('Output');
% legend('Function','Training Data','Approximated Function','Location','northoutside','Orientation','horizontal');
% 
% title(['I/p Vs O/p for  M(minimum Validation Error) = ' num2str(minIndex)]);
% axis([0 1 0 4]);
% hold off;
% 
% h = figure
% plot(0:14,TrainERMS,'-r','LineWidth',2);
% hold on;
% plot(0:14,TestERMS,'-g','LineWidth',2);
% plot(0:14,ValERMS,'-b','LineWidth',2);
% xlabel('Model Complexity M')
% ylabel('ERMS)');
% axis([0 14 0 1])
% legend('Train Data','Test Data','Validation Data','Location','northoutside','Orientation','horizontal');
% saveas(h, ['ERMS 501 samples.fig']);


% 
%plotting for minimum Train ERMS and Test ERMS 
%find M
[~,ind] = min(TrainERMS);

[~,ind1] = min(TestERMS);
[~,ind2] = min(ValERMS);
% 
% %whole program is for 0 : 15
% %regularization for  M = 14

clear TrainERMS ValERMS TestERMS
phi = phi14;
sPHI = phi'*phi;
DIM = size(sPHI,2);
lambdaArray = [1,exp(-0.5),exp(-1),exp(-1.5),exp(-2),exp(-2.25),exp(-3),exp(-3.5),exp(-4),exp(-4.5),exp(-5),exp(-5.5),exp(-6),exp(-6.5),exp(-7),exp(-7.5),exp(-8),exp(-9),exp(-10),exp(-10.5),exp(-11),exp(-11.5),exp(-12),exp(-13),exp(-14),exp(-15),exp(-16),exp(-17),exp(-18)];
lnLambda = log(lambdaArray);
lnLam = [0,-10.5,-15,-18];
TrainERMS = [];
TestERMS = [];
ValERMS = [];
weightCell = cell(1,size(lnLam,2));
jk = 1;
for lamIter = 1 : size(lambdaArray,2);
    lambda =lambdaArray(lamIter) * eye(DIM);
    W = inv(lambda + sPHI)*phi'*Target';
    if lnLambda(lamIter) == 0
        WSTORE = W
    end
    if lnLambda(lamIter) == -10.5
        WSTORE1 = W
    end
    
    if lnLambda(lamIter) == -18
        WSTORE2 = W
    end
    ModelOutput = zeros(1,size(X,2)); %change to X for 21 samples
    M = 14;  %change to 14 for model complexity 14
    for i = 1 : size(X,2)
        ModelOutput(i) = 0;
        for m = 1 : M+1
            ModelOutput(i) = ModelOutput(i) + W(m)*(X(i)^(m-1));
        end
    end
    
    %calulate TrainERMS
ModelTrain = ModelOutput(TrainInd);
error = sum((Target - ModelTrain).*(Target - ModelTrain)) + lambdaArray(lamIter)*(W'*W);
TrainERMS(lamIter) = sqrt(error/size(Target,2));

%calculate TestERMS
ModelTest = ModelOutput(TestIndices);
error = sum((testTarget - ModelTest).*(testTarget - ModelTest)) + lambdaArray(lamIter)*(W'*W);
TestERMS(lamIter) = sqrt(error/size(testTarget,2));

ModelVal = ModelOutput(ValIndices);
error = sum((valTarget - ModelVal).*(valTarget - ModelVal)) + lambdaArray(lamIter)*(W'*W);
ValERMS(lamIter) = sqrt(error/size(valTarget,2));
    
    if ismember(lnLambda(lamIter),lnLam) == 1 % doing scatter plot and model vs target for -3 -6 and -8
       h =  figure;
        plot(Input,Target,'or'); 
        hold on;
        plot(Input,ModelTrain,'ob');
        title(['Target Output Vs Model Output for Training data for ln\lambda = ' num2str(lnLambda(lamIter))]);
        xlabel('Input');
        ylabel('Output');
        plot(X,Y,'-g','LineWidth',0.5);
        legend('Target Output','Model Output','Fuction');
        saveas(h, ['501 samples M = 14 TOMO  train' num2str(lnLambda(lamIter)) '.fig']); 
        weightCell{1,jk} = W;
        jk = jk + 1;
       h =  figure;
        scatter(Target,ModelTrain,'ob');
        hold on;
         title(['Scatter plot of Target Output Vs Model Output for Training data for ln\lambda = ' num2str(lnLambda(lamIter))]);
         xlabel('Target Output');
         ylabel('Model Output');
        plot(0:4,0:4,'-r');
        axis([0 4 0 4]);
           saveas(h, ['501 samples M = 14 Scatter  Train ' num2str(lnLambda(lamIter)) '.fig']);
        
       h = figure;
        plot(testInput,testTarget,'or'); 
        hold on;
        plot(testInput,ModelTest,'ob');
        title(['Target Output Vs Model Output for Testing data for ln\lambda = ' num2str(lnLambda(lamIter))]);
        xlabel('Input');
        ylabel('Output');
        axis([0 1 0 6]);
        plot(X,Y,'-g','LineWidth',0.5);
        legend('Target Output','Model Output','Fuction');
        saveas(h, ['501 samples M = 14 TOMO test' num2str(lnLambda(lamIter)) '.fig']); 
        
        h = figure;
        scatter(testTarget,ModelTest,'ob');
        hold on;
         title(['Scatter plot of Target Output Vs Model Output for Test data for ln\lambda = ' num2str(lnLambda(lamIter))]);
         xlabel('Target Output');
         ylabel('Model Output');
        plot(0:6,0:6,'-r');
        axis([0 6 0 6]);
        saveas(h, ['501 samples M = 14 Scatter  Test ' num2str(lnLambda(lamIter)) '.fig']);
        
        h = figure;
        plot(valInput,valTarget,'or'); 
        hold on;
        plot(valInput,ModelVal,'ob');
        title(['Target Output Vs Model Output for Validation data for ln\lambda = ' num2str(lnLambda(lamIter))]);
        xlabel('Input');
        ylabel('Output');
       plot(X,Y,'-g','LineWidth',0.5);
        legend('Target Output','Model Output','Fuction');
        saveas(h, ['501 samples M = 14 TOMO  validation' num2str(lnLambda(lamIter)) '.fig']);
        
       h = figure;
        plot(valTarget,ModelVal,'ob');
        hold on;
         title(['Scatter plot of Target Output Vs Model Output for Validation data for ln\lambda = ' num2str(lnLambda(lamIter))]);
         xlabel('Target Output');
         ylabel('Model Output');
        plot(0:4,0:4,'-r');
        axis([0 4 0 4]);
        
        saveas(h, ['501 samples M = 14 Scatter  Val ' num2str(lnLambda(lamIter)) '.fig']);
        
        
    end
    h = figure;
    plot (X,Y,'LineWidth',2);
    weightCell{lamIter,lamIter} = W;
    hold on;
    scatter (Input,Target,'o');
    hold on;
    plot (X,ModelOutput,'-r','LineWidth',2);
    xlabel('Input');
    ylabel('Output');
    legend('Function','Training Data','Approximated Function','Location','northoutside','Orientation','horizontal');
    
    title(['I/p Vs O/p for M = 14 ln \lambda = ' num2str(lnLambda(lamIter))]);
    axis([0 1 0 4]);
     saveas(h, ['501 samples M = 14 ip VS op   ' num2str(lnLambda(lamIter)) '.fig']);
    
    
    
    hold off;
end


figure
plot(lnLambda,TrainERMS,'-r','LineWidth',2);
hold on;
plot(lnLambda,TestERMS,'-g','LineWidth',2);
plot(lnLambda,ValERMS,'-b','LineWidth',2)
xlabel('ln(lambda)')
ylabel('ERMS');

legend('Train Data','Test Data','Validation Data','Location','northoutside','Orientation','horizontal');

[~,ind] = min(TrainERMS);

[~,ind1] = min(TestERMS);
[~,ind2] = min(ValERMS);

lnLambda(ind)
lnLambda(ind1)
lnLambda(ind2)
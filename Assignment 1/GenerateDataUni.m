% creating test data for 15 data points
 X = 0:0.001:1;
 
 Y = exp(tanh(2*pi*X));
 Y_n = Y + normrnd(0,0.1,1,size(Y,2));
 
 figure;
 plot(X,Y);
 hold on
 plot(X,Y_n,'or');

% Train = [X'  Y_n'];
% save Train_101

X_limit = size(X,2);



TrainSize = ceil(0.6 * X_limit);
ValSize = floor(0.25 * X_limit);
TrainInd = randperm( X_limit,TrainSize);

RemInd = setdiff(1:X_limit,TrainInd);
ValInd = randperm(size(RemInd,2),ValSize);
ValIndices = RemInd(ValInd);

TestIndices = setdiff(RemInd,ValIndices);


Target = Y_n(TrainInd);
Input = X(TrainInd);          %X y_new for 21 sample X_new and Y_new for larger samples

testTarget = Y_n(TestIndices);
testInput = X(TestIndices);

valTarget = Y_n(ValIndices);
valInput  = X(ValIndices);

figure;
plot(testInput,testTarget,'ob');
 hold on
plot(X,Y);

title('Test')

figure
plot(valInput,valTarget,'ob');
hold on
plot(X,Y);
title('Val');

figure
plot(Input,Target,'ob');
hold on
plot(X,Y);
title('Target');




 
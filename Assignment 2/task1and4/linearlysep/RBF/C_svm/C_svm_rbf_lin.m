clear;
clc;
class1_test=load('../../group6/class1_test.txt');
class1_train=load('../../group6/class1_train.txt');
class1_val=load('../../group6/class1_val.txt');

class2_test=load('../../group6/class2_test.txt');
class2_train=load('../../group6/class2_train.txt');
class2_val=load('../../group6/class2_val.txt');

class3_test=load('../../group6/class3_test.txt');
class3_train=load('../../group6/class3_train.txt');
class3_val=load('../../group6/class3_val.txt');

traindata=[class1_train;class2_train;class3_train];
valdata=[class1_val;class2_val;class3_val];
testdata=[class1_test;class2_test;class3_test];

labelstrain=[ones(250,1)*[1,0,0];ones(250,1)*[0,1,0];ones(250,1)*[0,0,1]];
labelstrain=vec2ind(labelstrain')';

labelsval=[ones(150,1)*[1,0,0];ones(150,1)*[0,1,0];ones(150,1)*[0,0,1]];
labelsval=vec2ind(labelsval')';

labelstest=[ones(100,1)*[1,0,0];ones(100,1)*[0,1,0];ones(100,1)*[0,0,1]];
labelstest=vec2ind(labelstest')';

totaldata=[class1_train;class1_val;class1_test;class2_train;class2_val;class2_test;class3_train;class3_val;class3_test];
labelsdata=[ones(500,1)*[1,0,0];ones(500,1)*[0,1,0];ones(500,1)*[0,0,1]];
labelsdata=vec2ind(labelsdata')';

model = svmtrain(labelstrain, traindata, '-s 0 -t 2 -d 3 -g 0.002 -r 1 -c 1 -n 0.5');

train_labels=svmpredict(labelstrain,traindata,model);
confusion_train=confusionmat(labelstrain,train_labels);

val_labels=svmpredict(labelsval,valdata,model);
confusion_val=confusionmat(labelsval,val_labels);

test_labels=svmpredict(labelstest,testdata,model);
confusion_test=confusionmat(labelstest,test_labels);

total_labels=svmpredict(labelsdata,totaldata,model);
confusion_total=confusionmat(labelsdata,total_labels);
save('C_svm_rbf_lin');


SV=model.sv_indices;
[D_SV,~]=size(SV);

% set up the domain over which you want to visualize the decision
% boundary
xrange = [-20 20];
yrange = [-20 20];
% step size for how finely you want to visualize the decision boundary.
inc = 0.5;
 
% generate grid coordinates. this will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
 
% size of the (x, y) image, which will also be the size of the 
% decision boundary image that is used as the plot background.
image_size = size(x);
 
%xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.

xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];

%gridX = -20 : 0.5 : 20;
%gridY = -20 : 0.5: 20;
%[Xgrid,Ygrid] = meshgrid(gridX,gridY);
XgridNew = reshape(x,[size(x,1)*size(x,2),1]);
YgridNew = reshape(y,[size(y,1)*size(y,2),1]);
testDataGrid = [XgridNew YgridNew];
[D,~]=size(testDataGrid);
decision_labels=svmpredict(zeros(D,1),testDataGrid,model);


decisionmap = reshape(decision_labels, image_size);
H=figure;
 
%show the image
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
 
% colormap for the classes:
% class 1 = light red, 2 = light green, 3 = light blue
cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
colormap(cmap);

plot(class1_train(:,1),class1_train(:,2), 'r.');
plot(class2_train(:,1),class2_train(:,2), 'go');
plot(class3_train(:,1),class3_train(:,2), 'b*');
 

legend('Class 1', 'Class 2', 'Class 3','Location','NorthOutside', ...
    'Orientation', 'horizontal');



for j=1:D_SV
    if(labelstrain(SV(j))==1)
        h1=plot(traindata(SV(j),1),traindata(SV(j),2),'ms');
    end
    if(labelstrain(SV(j))==2)
        h2=plot(traindata(SV(j),1),traindata(SV(j),2),'ks');
    end
    if(labelstrain(SV(j))==3)
        h3=plot(traindata(SV(j),1),traindata(SV(j),2),'cs');
    end
end
 
title('Decision Region');
xlabel('Feature 1');
ylabel('Feature 2');
saveas(H,'DB_C_rbf_lin.png');
close(H);

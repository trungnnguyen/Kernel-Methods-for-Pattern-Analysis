clear;
clc;

xMin=-4;
xMax=4;
yMin=-4;
yMax=4;

gridX = xMin : 0.1 : xMax;
gridY = yMin : 0.1: yMax;
[Xgrid,Ygrid] = meshgrid(gridX,gridY);
XgridNew = reshape(Xgrid,[size(Xgrid,1)*size(Xgrid,2),1]);
YgridNew = reshape(Ygrid,[size(Ygrid,1)*size(Ygrid,2),1]);
testDataGrid = [XgridNew YgridNew];

load('m4');
iw=net.IW{1};
iwb=net.b{1};
hw1=net.LW{2,1};
hw1b=net.b{2};
hw2=net.LW{3,2};
hw2b=net.b{3};

x1=testDataGrid;
y=y';
[d,~]=size(x1);
%%%%%1hw%%%%%
a1=x1*iw'+ones(d,1)*iwb';
outputhw1=tansig(a1);
x2=outputhw1;
%%%%%%%%2%%%%%%%%%%%
a2=x2*hw1'+ones(d,1)*hw1b';
outputhw2=tansig(a2);
x3=outputhw2;
%%%%%%%%%%3%%%%%%%%%%%
a3=x3*hw2'+ones(d,1)*hw2b';
ytest=tansig(a3);
ytestind=vec2ind(ytest');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%modelGrid=x3';
modelGrid = net(testDataGrid');
modelGridNew1 = reshape(modelGrid(1,:),[size(Xgrid,1),size(Ygrid,1)]);
modelGridNew2 = reshape(modelGrid(2,:),[size(Xgrid,1),size(Ygrid,1)]);
modelGridNew3 = reshape(modelGrid(3,:),[size(Xgrid,1),size(Ygrid,1)]);
%modelGridNew4 = reshape(modelGrid(4,:),[size(Xgrid,1),size(Ygrid,1)]);

figure;
hold on;
set(gca,'fontsize',18);
%surf(Xgrid,Ygrid,modelGridNew1);
%surf(Xgrid,Ygrid,modelGridNew2);
surf(Xgrid,Ygrid,modelGridNew3);
%surf(Xgrid,Ygrid,modelGridNew4);
title('output of 3rd node of Output Layer');




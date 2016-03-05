xMin=min(x);
xMax=max(x);
yMin=min(y);
yMax=max(y);
gridX = xMin : 0.2 : xMax;
gridY = yMin : 0.2: yMax;
[Xgrid,Ygrid] = meshgrid(gridX,gridY);
XgridNew = reshape(Xgrid,[size(Xgrid,1)*size(Xgrid,2),1]);
YgridNew = reshape(Ygrid,[size(Ygrid,1)*size(Ygrid,2),1]);
testDataGrid = [XgridNew YgridNew];
modelGrid = net(testDataGrid');
modelGridNew = reshape(modelGrid,[size(Xgrid,1),size(Ygrid,1)]);
figure;
surf(Xgrid,Ygrid,modelGridNew);
hold on;
scatter3(targetTrain(:,1),targetTrain(:,2),targetTrain(:,3),15,[0 1 0],'filled');
title('output of output node');
colormap winter
color bar
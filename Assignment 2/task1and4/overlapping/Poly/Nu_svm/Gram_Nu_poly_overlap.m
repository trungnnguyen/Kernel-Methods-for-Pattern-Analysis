clear;
clc;
str='Nu_svm_poly_overlap';
load(strcat(str,'.mat'));
saveas(plotconfusion(ind2vec(labelsdata'),ind2vec(total_labels')),strcat(str,'cm_all.png'));
saveas(plotconfusion(ind2vec(labelstest'),ind2vec(test_labels')),strcat(str,'cm_test.png'));
saveas(plotconfusion(ind2vec(labelstrain'),ind2vec(train_labels')),strcat(str,'cm_train.png'));
saveas(plotconfusion(ind2vec(labelsval'),ind2vec(val_labels')),strcat(str,'cm_val.png'));
close();
[D,~]=size(traindata);

a=1000;
b=0.002;
d=3;

KGTrain=kernalGram(traindata,traindata,'poly',a,b,d);
KGVal=kernalGram(valdata,valdata,'poly',a,b,d);
KGTest=kernalGram(testdata,testdata,'poly',a,b,d);

KGTrain=(KGTrain-min(min(KGTrain)))/(max(max(KGTrain))-min(min(KGTrain)));
KGTest=(KGTest-min(min(KGTest)))/(max(max(KGTest))-min(min(KGTest)));
KGVal=(KGVal-min(min(KGVal)))/(max(max(KGVal))-min(min(KGVal)));

[Dtrain,~]=size(traindata);
[Dval,~]=size(valdata);
[Dtest,~]=size(testdata);

for i=1:Dtrain
    for j=1:Dtrain
        KGTrain(i,j)=KGTrain(i,j)/sqrt(KGTrain(i,i)*KGTrain(j,j));
    end
end

for i=1:Dval
    for j=1:Dval
        KGVal(i,j)=KGVal(i,j)/sqrt(KGVal(i,i)*KGVal(j,j));
    end
end

for i=1:Dtest
    for j=1:Dtest
        KGTest(i,j)=KGTest(i,j)/sqrt(KGTest(i,i)*KGTest(j,j));
    end
end

[C1train,~]=size(class1_train);
[C2train,~]=size(class2_train);
[C3train,~]=size(class3_train);
[C4train,~]=size(class4_train);

[C1test,~]=size(class1_test);
[C2test,~]=size(class2_test);
[C3test,~]=size(class3_test);
[C4test,~]=size(class4_test);

[C1val,~]=size(class1_val);
[C2val,~]=size(class2_val);
[C3val,~]=size(class3_val);
[C4val,~]=size(class4_val);

G=KGTrain;
C=[C1train,C2train,C3train,C4train];
numSamples=Dtrain;
ideal = zeros(numSamples,numSamples);

    [~,N]=size(C);
    init1 = 1;init2 = 1;
    for i = 1:N
        s1 = C(i);
        s2 = C(i);
        ideal(init1:s1+init1-1,init2:s2+init2-1)=1;
        init1 = init1+s1; init2 = init2+s2;
        
    end
    
imwrite(ideal,'ideal.png');

error = norm(ideal-G,'fro')

strout=strcat(str,'.png');
figure();
imshow(G);
imwrite(G,strout);
close();
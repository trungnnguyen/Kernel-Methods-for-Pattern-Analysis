clear all;
close all;

lmin=-25;
lmax=10;
Tfull =[20 100 1000 2000];

for tt=Tfull
    trainsize=tt;
    X = load(['group6_train',num2str(trainsize),'.txt']);
    V = load('group6_val.txt');
    T = load('group6_test.txt');

    valsize=trainsize*0.15;
    testsize=trainsize*0.1;
    V=V(1:valsize,:);
    T=T(1:testsize,:);
    [row,col]=size(X);
    s=mean(sqrt(var(X(:,1:2))));

    Mfull=1:1:60;
    indices = find(Mfull>trainsize);
    Mfull(indices) = [];
    for KK=Mfull
        K=KK;
        M=K;
        [idx,C,sumd,D1] = kmeans(X(:,1:2),K);
        for i=1:row
            for j=1:K
                D=sum((X(i,1:2)-C(j,:)).^2);
                phi(i,j)= exp(- D/(2*s));
            end
        end
        %train
        Wml=(inv(phi'*phi))*phi'*X(:,3);
        %Wml=pinv(phi)*X(:,3);

        for i=1:row
           ytrain(i)= Wml'*phi(i,:)';
        end
        acc(K) = 0.5 * sum((ytrain'-X(:,3)).^2);
        Ermstrain(K)= sqrt((2*acc(K))/row);

        %val
        clear y;
        for i=1:valsize
            for j=1:K
                D=sum((V(i,1:2)-C(j,:)).^2);
                phi(i,j)= exp(-D/(2*s));
            end
        end
        for i=1:valsize
           yval(i)= Wml'*phi(i,:)';
        end
        acc = 0.5 * sum((yval'-V(:,3)).^2);
        Ermsval(K)= sqrt((2*acc)/valsize);

        %test
        clear y;
        for i=1:testsize
            for j=1:K
                D=sum((T(i,1:2)-C(j,:)).^2);
                phi(i,j)= exp(-D/(2*s));
            end
        end
        for i=1:testsize
           ytest(i)= Wml'*phi(i,:)';
        end
        acc = 0.5 * sum((ytest'-T(:,3)).^2);
        Ermstest(K)= sqrt((2*acc)/testsize);

                %  %plot scatter actual vs model op 3d
                % h = figure;
                % scatter3(X(:,1),X(:,2),X(:,3),'b');
                % hold on;
                % scatter3(X(:,1),X(:,2),ytrain','r');
                % xlabel('x:DataPoints');
                % ylabel('y:DataPoints');
                % zlabel('t:target value & y:model output');
                % title(['Target Output (blue) and Model Output (red) for M = ',num2str(M)]);
                % legend(['Train Data N = ' ,num2str(trainsize)]);
                % saveas(h, ['outputwithout/targetvsmodel train=' num2str(trainsize) 'M=' num2str(M) '.fig']);
                % saveas(h, ['outputwithout/targetvsmodel train=' num2str(trainsize) 'M=' num2str(M)  '.png']);
                % %close h;

                % %train 2d
                % h = figure;
                % scatter(X(:,3),ytrain','b');
                % xlabel('x:Target output');
                % ylabel('y:Model output');
                % title(['Scatter plot of train data for M = ',num2str(M)]);
                % legend(['Train Data N = ' ,num2str(trainsize)]);
                % saveas(h, ['outputwithout/trainscatter train=' num2str(trainsize) 'M=' num2str(M)  '.fig']);
                % saveas(h, ['outputwithout/trainscatter train=' num2str(trainsize) 'M=' num2str(M)  '.png']);
                % %val 2d
                % h = figure;
                % scatter(V(:,3),yval','g');
                % xlabel('x:Target output');
                % ylabel('y:Model output');
                % title(['Scatter plot of validation data for M = ',num2str(M)]);
                % legend(['Train Data N = ' ,num2str(trainsize)]);
                % saveas(h, ['outputwithout/valscatter train=' num2str(trainsize) 'M=' num2str(M) '.fig']);
                % saveas(h, ['outputwithout/valscatter train=' num2str(trainsize) 'M=' num2str(M) '.png']);
                % %test 2d
                % h = figure;
                % scatter(T(:,3),ytest','r');
                % xlabel('x:Target output');
                % ylabel('y:Model output');
                % title(['Scatter plot of test data for M = ',num2str(M)]);
                % legend(['Train Data N = ' ,num2str(trainsize)]);
                % saveas(h, ['outputwithout/valscatter train=' num2str(trainsize) 'M=' num2str(M)  '.fig']);
                % saveas(h, ['outputwithout/valscatter train=' num2str(trainsize) 'M=' num2str(M)  '.png']);

                % close all;

    end
    %scatter(y,X(:,3));


    h=figure;
    plot(Ermstrain,'b');
    hold on;
    plot(Ermsval,'g');
    hold on;
    plot(Ermstest,'r');
    xlabel('M : Model Complexity');
    ylabel('Erms : Root Mean Square Error');
    title('Change in Erms for various M values');
    legend(['Train Data N = ' ,num2str(trainsize)],'Validation Data','Test Data');
    saveas(h, ['outputwithout/ermsplot'  num2str(trainsize) '.fig']);
    saveas(h, ['outputwithout/ermsplot'  num2str(trainsize) '.png']);

end
% 
% 
% h = scatter3(T(:,1),T(:,2),T(:,3),'b');
% xlabel('x:DataPoints');
% ylabel('y:DataPoints');
% zlabel('t:target value');
% title('Scatter plot for training Data');
% legend(['Train Data N = ' ,num2str(trainsize)]);
% saveas(h, ['trainingdatascatter'  num2str(trainsize) '.fig']);

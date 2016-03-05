clear all;
close all;

lmin=-35;
lmax=10;

% larray =[-25 -6 1 -12 -7 -24];
larray =[-25];
Tfull=[2000];

for tt=Tfull
    trainsize=tt;
    X = load(['group6_train',num2str(trainsize),'.txt']);
    V = load('group6_val.txt');
    T = load('group6_test.txt');
    
    % h = scatter3(X(:,1),X(:,2),X(:,3),'b');
    % xlabel('x:DataPoints');
    % ylabel('y:DataPoints');
    % zlabel('t:target value');
    % title('Scatter plot for training Data');
    % legend(['Train Data N = ' ,num2str(trainsize)]);
    % saveas(h, ['outputs1/trainingdatascatter'  num2str(trainsize) '.fig']);

    valsize=trainsize*0.15;
    testsize=trainsize*0.1;
    V=V(1:valsize,:);
    T=T(1:testsize,:);
    [row,col]=size(X);
    s=mean(sqrt(var(X(:,1:2))));

    Mfull=[5 12 26 35 46];
    %Mfull=[46];

    indices = find(Mfull>trainsize);
    Mfull(indices) = [];
    
    for K=Mfull
        M=K;
        [idx,C,sumd,D] = kmeans(X(:,1:2),K);
        for i=1:row
            for j=1:K
                D=sum((X(i,1:2)-C(j,:)).^2);
                phi(i,j)= exp(-D/(2*s));
            end
        end

        for loglambda=lmin:lmax
  %           loglambdafull=larray
		% loglambda=loglambdafull;
            clear ll;
            ll= exp(loglambda);
            %train
            Wml=(inv((phi'*phi)+(ll*eye(K)))*phi'*X(:,3));

            for i=1:row
               ytrain(i)= Wml'*phi(i,:)';
            end
            acc = 0.5 * sum((ytrain'-X(:,3)).^2) + (ll/2)*(norm(Wml).^2) ;
            Ermstrain(loglambda-lmin+1)= sqrt((2*acc)/row);

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
            acc = 0.5 * sum((yval'-V(:,3)).^2) + (ll/2)*(norm(Wml).^2) ;
            Ermsval(loglambda-lmin+1)= sqrt((2*acc)/valsize);

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
            acc = 0.5 * sum((ytest'-T(:,3)).^2) + (ll/2)*(norm(Wml).^2) ;
            Ermstest(loglambda-lmin+1)= sqrt((2*acc)/testsize);

            % %plot scatter actual vs model op 3d
            % h = figure;
            % scatter3(X(:,1),X(:,2),X(:,3),'b');
            % hold on;
            % scatter3(X(:,1),X(:,2),ytrain','r');
            % xlabel('x:DataPoints');
            % ylabel('y:DataPoints');
            % zlabel('t:target value & y:model output');
            % title(['Target Output (blue) and Model Output (red) for M = ',num2str(M),' & ln(lamda) = ',num2str(loglambda)]);
            % legend(['Train Data N = ' ,num2str(trainsize)]);
            % saveas(h, ['outputs1/targetvsmodel train=' num2str(trainsize) 'M=' num2str(M) 'loglamda = ' num2str(loglambda) '.fig']);
            % saveas(h, ['outputs1/targetvsmodel train=' num2str(trainsize) 'M=' num2str(M) 'loglamda = ' num2str(loglambda) '.png']);
            % %close h;
            
            % %train 2d
            % h = figure;
            % scatter(X(:,3),ytrain','b');
            % xlabel('x:Target output');
            % ylabel('y:Model output');
            % title(['Scatter plot of train data for M = ',num2str(M),' & ln(lamda) = ',num2str(loglambda)]);
            % legend(['Train Data N = ' ,num2str(trainsize)]);
            % saveas(h, ['outputs1/trainscatter train=' num2str(trainsize) 'M=' num2str(M) 'loglamda = ' num2str(loglambda) '.fig']);
            % saveas(h, ['outputs1/trainscatter train=' num2str(trainsize) 'M=' num2str(M) 'loglamda = ' num2str(loglambda) '.png']);
            % %val 2d
            % h = figure;
            % scatter(V(:,3),yval','g');
            % xlabel('x:Target output');
            % ylabel('y:Model output');
            % title(['Scatter plot of validation data for M = ',num2str(M),' & ln(lamda) = ',num2str(loglambda)]);
            % legend(['Validation Data N = ' ,num2str(trainsize)]);
            % saveas(h, ['outputs1/valscatter train=' num2str(trainsize) 'M=' num2str(M) 'loglamda = ' num2str(loglambda) '.fig']);
            % saveas(h, ['outputs1/valscatter train=' num2str(trainsize) 'M=' num2str(M) 'loglamda = ' num2str(loglambda) '.png']);
            % %test 2d
            % h = figure;
            % scatter(T(:,3),ytest','r');
            % xlabel('x:Target output');
            % ylabel('y:Model output');
            % title(['Scatter plot of test data for M = ',num2str(M),' & ln(lamda) = ',num2str(loglambda)]);
            % legend(['Test Data N = ' ,num2str(trainsize)]);
            % saveas(h, ['outputs1/testscatter train=' num2str(trainsize) 'M=' num2str(M) 'loglamda = ' num2str(loglambda) '.fig']);
            % saveas(h, ['outputs1/testscatter train=' num2str(trainsize) 'M=' num2str(M) 'loglamda = ' num2str(loglambda) '.png']);
            
            % close all;
        end

        %model vs target scatter plot
        h1=figure;
        plot(lmin:1:lmax,Ermstrain,'b');
        hold on;
        plot(lmin:1:lmax,Ermsval,'g');
        hold on;
        plot(lmin:1:lmax,Ermstest,'r');
        hold on;
        title(['Change in Erms for various values of ln(lamda) for M =',int2str(M)]);
        xlabel('ln(lamda) Regularization Parameter');
        ylabel('Erms : Root Mean Square error');
        legend(['Train Data N=',int2str(trainsize)],'Validation Data','Test Data');
        saveas(h1, ['outputs1/Ermswithlamda train=' num2str(trainsize) 'M=' num2str(M) '.fig']);
        saveas(h1, ['outputs1/Ermswithlamda train=' num2str(trainsize) 'M=' num2str(M) '.png']);
        %close h1;
        close all;
        
        
    end
end





%plot(y,T(:,3));

function e = computeError(w,Data,target,CTrain,width,logLambda,phiTilda)
    %calculating phi for test data or valiation data 
    %CTrain gives the centroids for that particular model we have to
    %calulcate phi
    rows = size(Data,1);
    columns = size(CTrain,1);
    phi = zeros(rows,columns);
    for r = 1 : rows
        for c = 1 : columns
            dist = (Data(r,:) - CTrain(c,:))*(Data(r,:) - CTrain(c,:))';
            phi(r,c) = exp(-dist/width);
        end
        
    end
    model = phi*w;
    e = sum((target - model).*(target - model))/2 +  (exp(logLambda)/2)*w'*phiTilda*w;
    e = sqrt(2*e/size(model,1));
    
end
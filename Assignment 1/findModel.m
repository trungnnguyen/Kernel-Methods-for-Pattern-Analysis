function model = findModel(w,Data,optCentroids,width)
%calculating phi for test data or valiation data 
    %CTrain gives the centroids for that particular model we have to
    %calulcate phi
    rows = size(Data,1);
    columns = size(optCentroids,1);
    phi = zeros(rows,columns);
    for r = 1 : rows
        for c = 1 : columns
            dist = (Data(r,:) - optCentroids(c,:))*(Data(r,:) - optCentroids(c,:))';
            phi(r,c) = exp(-dist/width);
        end
        
    end
    model = phi*w;
end
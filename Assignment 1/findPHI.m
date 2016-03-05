function phi = findPHI(D,width,C,Data)
    phi = zeros(size(D));
    for r = 1 : size(D,1)
        for c = 1 : size(D,2)
            %phi(r,c) = exp(-(D(r,c)^2)/width);
            
            %%calculating distance of each point from centriod
            dist = (Data(r,:)-C(c,:))*(Data(r,:)-C(c,:))';
            phi(r,c) = exp(-dist/width);
        end
    end
end
function phiTilda = findPHITILDA(C,width)
phiTilda = zeros(size(C,1));
for i = 1 : size(C,1)
    for j = 1 :size(C,1)
        phiTilda(i,j) = exp(-((C(i,:)-C(j,:))*(C(i,:)-C(j,:))')/width);
    end
end
end
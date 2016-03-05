function w = findWeights(phi,phiTilda,trainDataTarget,lambda)

    w = pinv(phi'*phi + exp(lambda)*phiTilda)*phi'*trainDataTarget;
end
function alpha = computeAlpha(C)
    n = size(C,1);
    alpha = -1;
    for i = 1 : n
        for j = 1 : n
            dist = norm((C(i,:)-C(j,:)),2);
            if alpha < dist
                alpha = dist;
            end
        end
    end
    alpha = alpha*alpha;
end
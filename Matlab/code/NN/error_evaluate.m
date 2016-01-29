function [error, acc, t, y] = error_evaluate(net, x, t, mode, threshold)
    
    error = 0;
    
    y = mlpfwd(net, x);
    
    for r = 1:net.nout
        error = error + t(:,r)'*log(y(:,r));             
    end

    error = -error/size(y,1);
    
    if strcmp(mode,'multi-label') == 1
        if exist('threshold') == 0
            %threshold = [0.22, 0.18, 0.2, 0.18, 0.18];
            threshold = repmat(0.2, 1, size(t, 2));
        end
        res = [y >= repmat(threshold, size(y,1), 1)];
        for cl = 1:size(res,2)
            ans = ((t(:, cl) == res(:, cl)) & (t(:, cl) == 1));
            acc(1, cl) = sum(ans)/sum((t(:, cl) == 1));
            ans = ((t(:, cl) == res(:, cl)) & (t(:, cl) == 0));
            acc(2, cl) = sum(ans)/sum((t(:, cl) == 0));
        end
    elseif strcmp(mode,'binary') == 1
        y1 = (y>0.23);
        acc = sum((y1 == t))/size(y,1);
        ind = t;
        ind1 = y;
        p = zeros(2,2);
        p(1,1) = size(find(t(find(y1==t))==0),1);
        p(1,2) = size(find(t(find(y1~=t))==0),1);
        p(2,2) = size(find(t(find(y1==t))==1),1);
        p(2,1) = size(find(t(find(y1~=t))==1),1);
        p(1,1)/(p(1,1)+p(1,2))
        p(2,2)/(p(2,1)+p(2,2))
    elseif strcmp(mode,'multi-class') == 1
        [~, ind] = max(y');
        ind = ind';
        [~, ind1] = max(t');
        ind1 = ind1';
        p = zeros(1,net.nout);
        n = size(x);
        for i=1:n(1)
            p(ind1(i)) = p(ind1(i)) + [ind(i) == ind1(i)];
        end
        acc = sum(p)/n(1);
    end
end
function [dataset] = balancing_dataset(dataset)
    

    for i = min(dataset(:,1)):max(dataset(:,1))
        b{i} = find(dataset(:,1) == i);
        s(i) = size(b{i},1);
    end
    
    n = max(s);
    
    for i = 1:max(dataset(:,1))
       k = n - s(i);
       t = [];
        while size(b{i},1) < k
            t = [t; b{i}]; 
            k = k - size(b{i},1);
        end
        [q,~,~,~] = devide_data(b{i},b{i},k);
        q = [t; q];
        dataset = [dataset; dataset(q,:)];
    end
    
    p = randperm(size(dataset,1));
    dataset = dataset(p,:);
    a = ones(size(dataset,1),1);
	
    
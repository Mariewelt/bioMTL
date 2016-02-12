function [unit, i_opt, j_opt, m] = greedy_prunning_criteria(net, xlearn, ylearn, xcontrol, ycontrol)  

e1 = zeros(net.nin, net.nhidden);
e2 = zeros(net.nhidden, net.nout);

i_1 = 0;
j_1 = 0;
min_1 = inf;

for i=1:net.nin
    for j=1:net.nhidden 
        if net.a1(i,j)~= 0 
            w = net.w1(i,j); 
            net.w1(i,j) = 0; 
            net.a1(i,j) = 0; 
            net = train(net, xlearn, ylearn);
            auc = zeros(1, net.nout);
            [~, ~, classes, func_results] = error_evaluate(net, xcontrol, ycontrol, 'multi-label');
            for k = 1:net.nout
                [y1, y2] = remove999(classes, func_results, k);
                [~, ~, auc(k)] = roc_func(y1, y2, k);
            end
            auc
            e1(i, j) = min(auc);
            if e1(i, j) <= min_1
                min_1 = e1(i, j);
          		i_1 = i;
          		j_1 = j;
            end
            
            net.w1(i,j) = w;
        	net.a1(i,j) = 1;

        end
    end
end

min_2 = inf;
i_2 = 0;
j_2 = 0;

for i=1:net.nhidden
    for j=1:net.nout 
        if net.a2(i,j)~=0
            w = net.w2(i,j);
            net.w2(i,j) = 0;
            
            net = train(net, xlearn, ylearn);
            auc = zeros(1, net.nout);
            [~, ~, classes, func_results] = error_evaluate(net, xcontrol, ycontrol, 'multi-label');
            for k = 1:net.nout
                [y1, y2] = remove999(classes, func_results, k);
                [~, ~, auc(k)] = roc_func(y1, y2, k);
            end
            auc
            e2(i, j) = min(auc);
            net.w2(i,j) = w;
            if e2(i, j) <= min_2
                min_2 = e2(i, j);
            	i_2 = i;
         		j_2 = j;
            end
        end
    end
end

if (min_1 < min_2) && (i_1 ~= 0) && (j_1 ~= 0)
    unit = 1;
    i_opt = i_1;
    j_opt = j_1;
    m = i_opt*net.nhidden+j_opt;
end
if (min_2 < min_1) && (i_2 ~= 0) && (j_2 ~= 0)
    unit = 2;
    i_opt = i_2;
    j_opt = j_2;
    m = net.nin*net.nhidden + i_opt*net.nout+j_opt;
end

return
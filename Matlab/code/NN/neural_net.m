function [er, net] = neural_net(data, target, nhidden, ncycles)

nin = size(data,2);               % Number of inputs.
nout = size(target,2);               % Number of outputs.
if (nout ==1)
    net = mlp(nin, nhidden, nout, 'logistic');
else
    net = mlp(nin, nhidden, nout, 'logistic');
end
    
net.nps = (nin+nout) * nhidden;
net.w1 = zeros(nin, nhidden);
net.w2 = zeros(nhidden, nout);
net.a1 = ones(nin, nhidden); 
net.a2 = ones(nhidden, nout);
net.eta = 0.0002;
net.ncycles = ncycles;
net = train(net, data, target);
[~, er] = error_evaluate(net, data, target, 'multi-label');

end

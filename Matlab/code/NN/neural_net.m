function [er, net] = neural_net(data, target, nhidden, ncycles)

nin = size(data,2);               % Number of inputs.
nout = size(target,2);               % Number of outputs.
net = neural_net_init(nin, nout, nhidden, ncycles);
net = train(net, data, target);
[~, er] = error_evaluate(net, data, target, 'multi-label');

end

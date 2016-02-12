function net = neural_net_init(nin, nout, nhidden, ncycles)

net = mlp(nin, nhidden, nout, 'logistic');
    
net.nps = (nin+nout) * nhidden;
net.w1 = zeros(nin, nhidden);
net.w2 = zeros(nhidden, nout);
net.a1 = ones(nin, nhidden); 
net.a2 = ones(nhidden, nout);
net.eta = 0.0002;
net.ncycles = ncycles;

end

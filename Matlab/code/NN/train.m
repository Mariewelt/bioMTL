% This program trains the neural network using back-propagation algorithm
% for NCYCLES times with learning-rate parameter ETA

function net = train(net, xlearn, ylearn)

xbatch = xlearn;
ybatch = ylearn;
for i=1:net.ncycles
    [y, z] = mlpfwd(net, xbatch); 	%forward propagation
    a = (ybatch ~= 999);
    deltas = (y - ybatch).*a;
    g = mlpbkp_new(net, xbatch, z, deltas);
    ww = [net.w1(:)',net.b1, net.w2(:)', net.b2];
    net = mlpunpak(net, ww - net.eta*g);
end

end
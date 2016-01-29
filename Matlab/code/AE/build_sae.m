function [data, sae] = build_sae(data, target, sizes, numhid, numepochs, momentum)

%% train SAE and get output data

hiddenSize = zeros(1, sizes);
for i =1:sizes
    hiddenSize(i) = numhid;
end

numLayers = size(hiddenSize,2);
sparsityParam = momentum;   % desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term   
maxIter = numepochs;
inputSize = size(data,1);
numClasses = size(target,2);
target = target';
[~, ind] = max(target);
target = ind';

% Classificator training
%[stackedAETheta, netconfig] = training(inputSize, ...
stack = training(inputSize, ...
    hiddenSize, sparsityParam,...
    lambda, beta, maxIter, data);

% Extract out the "stack"
%stack = params2stack(stackedAETheta(hiddenSize(numLayers)*numClasses+1:end), netconfig);

depth = numel(stack);
a = {};
a{1} = data;
for layer = (1:depth)
    z{layer+1} = stack{layer}.w*a{layer} + repmat(stack{layer}.b,1,size(a{layer},2));
    a{layer+1} = sigmoid(z{layer+1});
end

data = a{depth+1};
sae.w = stack{1,1}.w;
sae.b = stack{1,1}.b;


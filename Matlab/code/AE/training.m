function stack = training(inputSize, ...
    hiddenSize, sparsityParam, lambda, beta, maxIter, trainData)

numLayers = size(hiddenSize,2);
%Training of the first sparse autoencoder

%  Randomly initialize the parameters

%sae1Theta = initializeParameters(hiddenSize(1), inputSize);

s = cell(numLayers,1);
s{1}.w = initializeParameters(hiddenSize(1), inputSize);
for i=2:numLayers
    s{i}.w = initializeParameters(hiddenSize(i), hiddenSize(i-1));
end

% Training of the first layer sparse autoencoder, this layer has
% an hidden size of "hiddenSizeL1"
% Optimal parameters stored in sae1OptTheta

options.Method = 'lbfgs'; 
options.maxIter = maxIter;	   
options.display = 'on';

sOpt = cell(numLayers,1);
[sOpt{1}.w, ~] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize(1), ...
                                   lambda, sparsityParam, ...
                                   beta, trainData), ...
                              s{1}.w, options);
sOpt{1}.f = feedForwardAutoencoder(sOpt{1}.w, hiddenSize(1), ...
                                        inputSize, trainData);
for i=2:numLayers
    [sOpt{i}.w, ~] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSize(i-1), hiddenSize(i), ...
                                   lambda, sparsityParam, ...
                                   beta, sOpt{i-1}.f), ...
                              s{i}.w, options);
                          

    sOpt{i}.f = feedForwardAutoencoder(sOpt{i}.w, hiddenSize(i), ...
                                        hiddenSize(i-1), sOpt{i-1}.f);
end

%%======================================================================
% Initialization of the stack using the parameters learned
stack = cell(numLayers,1);
stack{1}.w = reshape(sOpt{1}.w(1:hiddenSize(1)*inputSize), ...
                     hiddenSize(1), inputSize);
stack{1}.b = sOpt{1}.w(2*hiddenSize(1)*inputSize+1:2*hiddenSize(1)*inputSize+hiddenSize(1));

for i=2:numLayers
    stack{i}.w = reshape(sOpt{i}.w(1:hiddenSize(i)*hiddenSize(i-1)), ...
                     hiddenSize(i), hiddenSize(i-1));
    stack{i}.b = sOpt{i}.w(2*hiddenSize(i)*hiddenSize(i-1)+1:2*hiddenSize(i)*hiddenSize(i-1)+hiddenSize(i));
end
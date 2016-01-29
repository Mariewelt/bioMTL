function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; 
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));

depth = numel(stack);
z = cell(depth+1);
a = cell(depth+1);
d = cell(depth+1);
a{1} = data;
for layer = (1:depth)
    z{layer+1} = stack{layer}.w*a{layer} + repmat(stack{layer}.b,1,size(a{layer},2));
    a{layer+1} = sigmoid(z{layer+1});
end


x = softmaxTheta * a{depth+1};
M = max(x);
x = x - repmat(M, size(x,1),1);
x = exp(x);
NormDivider = repmat(ones(1,size(x,1)) * x,size(x,1),1);
x = x./NormDivider;
softmaxThetaGrad = -(a{depth+1} * (groundTruth - x)')'/numClasses + lambda*softmaxTheta;
p = x;
x = log(x);
cost = -(sum(sum(x.*groundTruth)))/numClasses;
cost = cost + (lambda/2)*(norm(softmaxTheta,'fro')^2);

d{depth+1} = -softmaxTheta' * (groundTruth - p) .* a{depth+1} .* (1 - a{depth+1});
for i = depth:-1:2
    d{i} = (stack{i}.w' * d{i+1}) .* a{i} .* (1 - a{i});
end

for i = 1:numel(stack)
    stackgrad{i}.w = (d{i+1} * a{i}') * (1/numClasses);
    stackgrad{i}.b = sum(d{i+1},2) * (1/numClasses);
end


%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data


theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

x = theta * data;
M = max(x);
x = x - repmat(M, size(x,1),1);
x = exp(x);
NormDivider = repmat(ones(1,size(x,1)) * x,size(x,1),1);
x = x./NormDivider;
thetagrad = -(data * (groundTruth - x)')'/numCases + lambda*theta;
x = log(x);
cost = -(sum(sum(x.*groundTruth)))/numCases + (lambda/2)*(norm(theta,'fro')^2);

grad = [thetagrad(:)];
end


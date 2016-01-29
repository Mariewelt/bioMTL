function [data, target, sae, xcontrol, ycontrol] = sae(filename, sizes, numhid, numepochs, momentum)
%% Read and preprocess data

dataset = read_data(filename);
dataset = dataset(randperm(size(dataset,1)),:);

y = dataset(:, 1);
X = dataset(:, 2: end);
X = mapstd(X');
X = X';

%[trainData, testData, trainLabels, testLabels] = devide_data(X, y, 4200);
[trainData, testData, trainLabels, testLabels] = devide_data(X, y, 120);
trainData = trainData';
testData = testData';
xcontrol = testData;
ycontrol = testLabels;

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
inputSize = size(dataset,2) - 1;
numClasses = max(max(dataset(:,1)));

% Classificator training
[stackedAETheta, stackedAEOptTheta, netconfig] = training(inputSize, ...
    numClasses, hiddenSize, sparsityParam, ...
    lambda, beta, maxIter, trainData, trainLabels);

% Extract out the "stack"
stack = params2stack(stackedAETheta(hiddenSize(numLayers)*numClasses+1:end), netconfig);

depth = numel(stack);
a{1} = trainData;
for layer = (1:depth)
    z{layer+1} = stack{layer}.w*a{layer} + repmat(stack{layer}.b,1,size(a{layer},2));
    a{layer+1} = sigmoid(z{layer+1});
end

data = a{depth+1};

target = trainLabels;
m = max(target);
target1 = zeros(size(target,1), m);
c = eye(m);
for i =1:size(target,1)
    target1(i,:) = c(target(i),:);
end

target = target1;

target1 = zeros(size(ycontrol,1), m);
for i =1:size(ycontrol,1)
    target1(i,:) = c(ycontrol(i),:);
end

ycontrol = target1;
sae = stack;

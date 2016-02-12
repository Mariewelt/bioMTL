clear;
clc;

%% Initialize all
% Initialize paths
addpath('code');
addpath('code\AE');
addpath('code\NN');
addpath('code\util');
addpath('data')

% Load data
load('dataset_file.mat');
load('fs.mat');

% Initialize parameters
classes = [1,2,3,5,12];
classes_num = size(classes, 2);

%[dataset, indexes] = prepare_data(features, y, classes);
%features(indexes, :) = [];
%y(indexes, :) = [];
%y(:,[5,6,7,8,9,10,11]) = [];

dataset = [y(:,classes), features];
[dataset, ps] = removeconstantrows(dataset');
validset = [y, features];
validset(:, ps.remove) = [];
dataset = dataset';
arr = sort(arr);
arr = arr(1:end-50);
dataset = dataset(:, [1:classes_num, arr+1+classes_num]);
filename2 = 'model.csv';
model_desc = read_data(filename2);

% Initialize classifier description
classifier_descr = struct('num_layers', 0, 'layer_type', [],...
    'numhid', [], 'numepochs',[]);

classifier_descr.num_layers = model_desc(1);
for i = 1:(model_desc(1))
    classifier_descr.layer_type(i) = model_desc(i+1);
    classifier_descr.numhid(i) = model_desc(i+1+model_desc(1));
    classifier_descr.numepochs(i) = model_desc(i+1+2*model_desc(1));
end

size_learn = round(size(dataset,1)*0.8);
cross_val_fold = 5;
ind_train = [];
ind_test = [];
mode = 'random';

%% Launch process
dataset = dataset(randperm(size(dataset,1)),:);
[xlearn, xcontrol, ylearn, ycontrol] = devide_data(dataset(:, classes_num+1:end), dataset(:,1:classes_num), size_learn);
[xlearn, ts] = mapstd(xlearn');
xlearn = xlearn';
xcontrol = mapstd('apply', xcontrol', ts);
xcontrol = xcontrol';

%[er_c, er_l, auc, model] = build_model(dataset, classifier_descr, size_learn, classes_num, ind_train, ind_test, mode);
%scores = cross_validation(dataset, classifier_descr, classes_num, cross_val_fold);

nin = size(xlearn, 2);
nout = classes_num;
nhidden = classifier_descr.numhid(1);
ncycles = classifier_descr.numepochs(1);

net = neural_net_init(nin, nout, nhidden, ncycles);

for i=1:100
    [unit, i_opt, j_opt, m] = greedy_prunning_criteria(net, xlearn, ylearn, xcontrol, ycontrol) 
end

% This function builds deep neural network of specified structure 
% CLASSIFIER_DESCR. DNN is trained using SIZE_LEARN objects, randomly 
% chosen from DATASET.
% INPUT:
% DATASET -- dataset of objects
% CLASSIFIER_DESCR -- classifier description
% SIZE_LEARN -- size of training dataset
% OUTPUT:
% ACC_C -- accuracy on a control dataset
% ACC_L -- accuracy on a training dataset
% AUC -- array with auc calues for each class
% MODEL -- structure which defines built model
 
function [acc_c, acc_l, auc, model] = build_model(dataset, classifier_descr, size_learn, classes, ind_train,ind_test, mode)

%% Randomly permutate the dataset
%dataset = dataset(randperm(size(dataset,1)),:);

y = dataset(:, 1:classes);
X = dataset(:, classes+1: end);


%% Randomly devide the dataset on train and test

if strcmp(mode, 'random') == 1
    [traindata, testData, trainLabels, testLabels] = devide_data(X, y, size_learn);
elseif strcmp(mode, 'fixed') == 1
    traindata = X(ind_train, :);
    testData = X(ind_test, :);
    trainLabels = y(ind_train, :);
    testLabels = y(ind_test, :);
else 
    error('Mode should be "random" or "fixed"');
end

%% Convert trainlabels into the demanded format
%m = max(testlabels);
%trainLabels = zeros(size(trainlabels,1), m);
%c = eye(m);
%for i =1:size(trainlabels,1)
%    trainLabels(i,:) = c(trainlabels(i),:);
%end
%testLabels = zeros(size(testlabels,1), m);
%c = eye(m);
%for i =1:size(testlabels,1)
%    testLabels(i,:) = c(testlabels(i),:);
%end

%if m==2
%    testLabels = testLabels(:,2);
%    trainLabels = trainLabels(:,2);
%end

%dataset = [trainlabels, traindata];
%dataset = balancing_dataset2(dataset);
%traindata = dataset(:,2:end);
%trainLabels = dataset(:,1);

%% Standartization of train and test datasets
[trainData, ts] = mapstd(traindata');
trainData = trainData';
testData = mapstd('apply', testData', ts);
testData = testData';
xcontrol = testData;
ycontrol = testLabels;

%% Initializing parameters for RBM and Autoencoder
sae_momentum = 0.1;
rbm_batchsize = 6;
rbm_momentum = 0.5;

%% Building model
data = trainData;
k = 0;
l = 0;
for i = 1:classifier_descr.num_layers
    if classifier_descr.layer_type(i) == 1
        k = k+1;
        [data, model.dbn{k}] = build_dbn(data, 1, classifier_descr.numhid(i), ...
            classifier_descr.numepochs(i), rbm_batchsize, rbm_momentum, k);   
        xcontrol = sigm(repmat(model.dbn{k}.rbm.b, size(xcontrol, 1), 1) + xcontrol * model.dbn{k}.rbm.W);    
    end
    if classifier_descr.layer_type(i) == 2
        data = data';
        l = l+1;
        [data, model.sae{l}] = build_sae(data, trainLabels, 1, classifier_descr.numhid(i), ...
            classifier_descr.numepochs(i), sae_momentum);
        layer = 1;
        a = {};
        a{1} = xcontrol';
        z{layer+1} = model.sae{layer,l}.w*a{layer} + repmat(model.sae{layer,l}.b,1,size(a{layer},2));
        a{layer+1} = sigmoid(z{layer+1});
        xcontrol = a{2};
        xcontrol = xcontrol';
        data = data';
    end     
    if classifier_descr.layer_type(i) == 3
        [data, q] = mapstd(data');
        data = data';
        xcontrol = mapstd('apply', xcontrol', q);
        xcontrol = xcontrol';
        [acc_l, model.net] = neural_net(data, trainLabels, classifier_descr.numhid(i), classifier_descr.numepochs(i));
    end
end

%% Finetuning model
%model = fine_tune(model, classifier_descr, trainData, trainLabels, 100);
%[y, z] = dnnfwd(model, classifier_descr, testData);
%xcontrol = z{end-2};

%% Evaluating errors and AUCs, making ROCs
[~, acc_c, classes, func_results] = error_evaluate(model.net, xcontrol, ycontrol, 'multi-label');

numclasses = size(ycontrol, 2);
%x_roc = zeros(size(ycontrol,1)+1, numclasses);
%y_roc = zeros(size(ycontrol,1)+1, numclasses);
auc = zeros(numclasses);

for i = 1:numclasses
    [y1, y2] = remove999(classes, func_results, i);
    [x_roc, y_roc, auc(i)] = roc_func(y1, y2, i);
    %fig = figure;
    %fig = ROC_plot(x_roc(:,i), y_roc(:,i));
    %num = num2str(i);
    %auc_s = num2str(auc(i));
    %annotation(fig,'textbox',...
    %[0.729079497907944 0.406801418439715 0.0826359832635983 0.0531914893617021],...
    %'String',{'AUC = ' auc_s});
    %roc_curve = ['report/' num '.png'];
    %saveas(fig, roc_curve, 'png');
    %close(fig);
end
auc = auc(:, 1)

end

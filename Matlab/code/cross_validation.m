function scores = cross_validation(dataset, classifier_descr, classes, fold)

%dataset = dataset(randperm(size(dataset,1)),:);
auc = zeros(fold, classes);
n = size(dataset, 1);
size_test = round(n/fold);
for i=1:fold
    start = (i-1)*size_test + 1;
    if (i*size_test) <= n
        stop = (i*size_test);
    else
        stop = n;
    end
    ind_test = start:stop;
    ind_train = 1:n;
    [~,idxsIntoA] = intersect(ind_train,ind_test);
    ind_train(idxsIntoA) = [];
    [~, ~, auc(i, :), ~] = build_model(dataset, classifier_descr, 0, classes, ind_train, ind_test, 'fixed');
    auc(i, :)
end

scores = auc;
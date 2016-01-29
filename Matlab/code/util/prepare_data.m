function [dataset, indexes_new] = prepare_data(features, y, l)
indexes = 1:size(features,1);
k = 0;

for i =1:max(indexes)
    if sum(y(i,l)) <= 12
        k = k+1;
        indexes_new(k) = i;
        features_new(k,:) = features(i,:);
        y_new(k,:) = y(i,l);
    end
end
 dataset = [y_new, features_new];
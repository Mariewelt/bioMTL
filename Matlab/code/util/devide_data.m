function [Learn, Test, y_learn, y_test]=devide_data(matr,y,size_learn)
%amount of objects received
col = size(y,1);
rvec=randperm(col);

Learn=matr(rvec(1,1:size_learn),:);
y_learn=y(rvec(1,1:size_learn),:);

Test=matr(rvec(1,(size_learn+1):col),:);
y_test=y(rvec(1,(size_learn+1):col),:);
return

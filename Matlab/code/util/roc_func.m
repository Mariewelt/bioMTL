function [x_roc, y_roc, auc] = roc_func(classes, func_results, cl_num)
% Gets a sample
% classes - represent real class marks
% func_results - represent the resultes of classificator used
%
% Returns ROC curve points and AUC
% x_roc - array of false positive rates
% y_roc - array of true positive rates, corresponding to x_roc
% auc - area under curve

%for i = 1:size(classes,1)
%    if func_results(i) == cl_num
%        func_results(i) = 1;
%    else
%        func_results(i) = 0;
%    end
%    if classes(i) == cl_num
%        classes(i) = 1;
%    else
%        classes(i) = 0;
%    end
%end

%%Sorting the sample by func_results in a descending order
  [func_results, indexes] = sort(func_results, 'descend');
  classes = classes(indexes);

%%
  l_pos=sum(classes==1);  %
  l_neg=sum(classes==0);  %
  l=size(func_results,1); %size of sample array
 
 %%Counting x_roc and y_roc 
  x_roc=zeros(l+1,1);
  y_roc=zeros(l+1,1);
  auc=0;
  for i=1:l
     if (classes(i,1)==0)
         x_roc(i+1,1)=x_roc(i,1)+1/l_neg;
         y_roc(i+1,1)=y_roc(i,1);
         auc=auc+1/l_neg*y_roc(i,1);
     else 
         x_roc(i+1,1)=x_roc(i,1);
         y_roc(i+1,1)=y_roc(i,1)+1/l_pos;
     end    
   end

end

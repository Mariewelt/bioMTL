function fig = ROC_plot(FPR, TPR)
%% Plots the ROC-curve
% Input (ROC-curve points):
%   FPR - false positive rates
%   TPR - true pasitive rates
% Output:
%   fig - the figure with a plot of ROC-curve

fig = figure;
hold on
set(gca, 'FontSize', 24, 'FontName', 'Times');
axis('tight');

xlabel('FPR','FontSize',24);
ylabel('TPR','FontSize',24);

plot(FPR,TPR,'r-','LineWidth',3);
plot([0;1],[0;1],'g-','LineWidth',3);

return
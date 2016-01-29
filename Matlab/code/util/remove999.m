function [y1, y2] = remove999(y1, y2, k)

i = 1;
y1 = y1(:, k);
y2 = y2(:, k);
while (i<=size(y1, 1))
    if y1(i) == 999
        y1(i) = [];
        y2(i) = [];
    else
        i = i+1;
    end
end
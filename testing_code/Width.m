function w = Width(x)
x1 = x(:,1:2:end);
x2 = x(:,2:2:end);
w = max(max(x1,[],2)-min(x1,[],2),max(x2,[],2)-min(x2,[],2))/100;
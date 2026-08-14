% Plot median with [5 95] confidence band in grey.

function plot_prctile(y)

% Inputs:
%   y: a ndraws-by-n matrix with each column representing a data series.
% Outputs:
%   plot a n-by-1 vector of median with [5 95] confidence band in grey shade.

n = size(y,2);
x = (1:n)';

yl = prctile(y, 5)';
ym = prctile(y, 50)';
yu = prctile(y, 95)';

close;
patch([x'  fliplr(x')],[yu'  fliplr(yl')],[0.7 0.7 0.7], 'EdgeColor','none');
hold on;
plot(x,ym,'b'); %color of median can be changed.
hold off;
% legend('90% bounds','median');
% Histograms of multiple data series

function hist_matrix(x)
% Inputs:
%     x: a n-by-K matrix with each column representing a data series.
% Outputs:
%     histograms of data series arranged in a two-column matrix form.

K = size(x,2);
m = K/2;
if m == round(m)
    for j = 1:K
        subplot(m,2,j);
        histogram(x(:,j),100);
        title(['para ',num2str(j)]);
    end
else
    m = (K+1)/2;
    for j = 1:K
        subplot(m,2,j);
        histogram(x(:,j),100);
        title(['para ',num2str(j)]);
    end
end

% Keep every k-th draw

function x2 = Thin(x1, k)
% Inputs:
%   x1: a n-by-m matrix of posterior draws
%   k: a scalar of the distance (e.g. 10)
% Outputs:
%   x2: a (n/k)-by-m matrix of the thinned draws (x(1), x(k+1), x(2k+1), ..., x((n-1)k+1)

[n,m] = size(x1);
h = n/k;
if h ~= round(n/k)
    disp(['n = ', num2str(n), ', k = ', num2str(k)]);
    error('n is not a multiple of k!');
end

x2 = zeros(h,m);
for i = 1:m
    x1i = x1(:,i);
    tmp = reshape(x1i, k, h);
    x2(:,i) = tmp(1,:)';
end
    
    
    


    




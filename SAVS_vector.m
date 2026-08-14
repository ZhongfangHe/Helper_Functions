% Sparsify b by SAVS (signal adaptive variable selector)
%
% if b > 1/||x||^(2/3), c = b - 1/(b * b * ||x||^2)
% if b < -1/||x||^(2/3), c = b + 1/(b * b * ||x||^2)
% otherwise, c = 0;


function cvec = SAVS_vector(bvec, x)
% Inputs:
%   bvec: a n-by-1 vector of raw coef estimate.
%   x: a ndraws-by-n matrix of regressors.
% Outputs:
%   cvec: a vector of sparsified coef estimate.

n = length(bvec);
cvec = zeros(n,1);
for j = 1:n
    xnorm2 = x(:,j)' * x(:,j);
    inv_xnorm23 = 1/(xnorm2^(1/3));
    
    b = bvec(j);
    if b >= inv_xnorm23
        c = b - 1/(b * b * xnorm2);
    elseif b <= -inv_xnorm23
        c = b + 1/(b * b * xnorm2);
    else
        c = 0;
    end
    cvec(j) = c;
end
    
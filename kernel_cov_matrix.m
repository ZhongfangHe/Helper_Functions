% Assemble kernel covariance matrix to used in "Est_TVL.m".
% K(i,j) = exp(-|xi-xj|^2)

function Kcov = kernel_cov_matrix(x)
% Inputs:
%   x: a n-by-m matrix of features.
% Outputs:
%   K: a n-by-n kernel covariance matrix

n = size(x,1);
Kcov = eye(n);
for i = 1:n-1
    for j = 2:n
        xi = x(i,:)';
        xj = x(j,:)';
        tmp1 = xi-xj;
        Kcov(i,j) = exp(-(tmp1'*tmp1));
        Kcov(j,i) = Kcov(i,j);
    end
end
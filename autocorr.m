% Compute autocorrelation for a matrix of data


function [x,y] = autocorr(z,k)
% Inputs:
%   z: a n-by-m matrix of data
%   k: a scalar of the max number of lags
% Outputs:
%   x: a k-by-1 vector of the lags
%   y: a k-by-m matrix of autocorrelations

[n,m] = size(z);
x = 1:k;
y = zeros(k,m);
for j = 1:k
    for jj = 1:m
        y(j,jj) = corr(z(1:(n-j),jj), z((1+j):n,jj));
    end
end
plot(x,y,'-o');
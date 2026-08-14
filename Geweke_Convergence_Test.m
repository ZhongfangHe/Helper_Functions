% Test convergence by Geweke (1992).
% Use the Newey-West method to estimate the long-run variance with the truncation lag floor(4 * (T/100)^(2/9)).
% Compute the effective sample size as another meric

function [gt_value, gt_cdf, ess] = Geweke_Convergence_Test(x)
% Inputs:
%   x: a n-by-m matrix of posterior draws (each column is for one parameter)
% Outputs:
%   gt_value: a m-by-1 vector of test value (following N(0,1) under null of converged)
%   gt_cdf: a m-by-1 vector of the CDF of test value under null (good if btw 0.025 and 0.975)
%   ess: a m-by-1 vector of the ESS for each column of x

[n,m] = size(x);

n1 = round(0.1*n);
n2 = round(0.5*n);

x1 = x(1:n1,:);
x2 = x((n-n2+1):n,:);

mu_x1 = mean(x1);
mu_x2 = mean(x2);
mu_x = mean(x);

u1 = x1 - ones(n1,1) * mu_x1;
u2 = x2 - ones(n2,1) * mu_x2;
u = x - ones(n,1) * mu_x;

k1 = floor(4 * ((n1/100)^(2/9)));
k2 = floor(4 * ((n2/100)^(2/9)));
k = floor(4 * ((n/100)^(2/9)));
% k = 100;
% k1 = floor(n1^(1/3));
% k2 = floor(n2^(1/3));
% k = 5000;%k = floor(n^(1/3));

var1 = var(x1);
var2 = var(x2);
varall = var(x);
for i = 1:m
    tmp = var1(i);
    for j = 1:k1
        tmp = tmp + 2 * (1 - j / (k1 + 1)) * u1(1:n1-j,i)' * u1(1+j:n1,i) / (n1-j);
    end
    var1(i) = tmp;
    
    tmp = var2(i);
    for j = 1:k2
        tmp = tmp + 2 * (1 - j / (k2 + 1)) * u2(1:n2-j,i)' * u2(1+j:n2,i) / (n2-j);
    end
    var2(i) = tmp;   
    
    tmp = varall(i);
    for j = 1:k
        tmp = tmp + 2 * (1 - j / (k + 1)) * u(1:n-j,i)' * u(1+j:n,i) / (n-j);
    end
    varall(i) = tmp;      
end

gt_value = (mu_x1 - mu_x2) ./ sqrt((var1 / n1 + var2 / n2));
gt_cdf = normcdf(gt_value);

% ess = var(x) ./ varall;
ess = effective_sample_size_portion(x)';

gt_value = gt_value';
gt_cdf = gt_cdf';
ess = ess';
    
    





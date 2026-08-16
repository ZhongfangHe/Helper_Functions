% Run OLS regression using backward elimination.
% yt = xt'*b + ut,
% ut is assumed to be serially uncorrelated but could be homoskedastic or heteroskedastic.
%
% The first regressor is the intercept that will always be kept.
% Non-constant regressors are kept only if their pvalues are small enough.


function est = Est_LinearReg_Stepwise(y,x,pvalue_star,ind_const_var)
% Inputs:
%   y: a n-by-1 vector of targets.
%   x: a n-by-m matrix of regressors (constant in the first column).
%   pvalue_star: a scalar of the threshold of pvalues (e.g. 0.05 for
%       5-percent significance level).
%   ind_const_var: a 0/1 indicator if the residual variance is constant.
% Outputs:
%   est: a structure with the following fields:
%     best: a m-by-1 vector of coef estimates (insignificant ones are zeros).
%     bstd: a m-by-1 vector of coef std errors (insignificant ones are zeros).
%     bzscore: a m-by-1 vector of coef z-scores (insignificant ones are zeros).
%     bpvalue: a m-by-1 vector of coef p-values (insignificant ones are zeros).
%     ind_sig: a m-by-1 vector of indicators of significant regressors.

[n,m] = size(x);
est.best = zeros(m,1);
est.bstd = zeros(m,1);
est.bzscore = zeros(m,1);
est.bpvalue = zeros(m,1);
est.ind_sig = zeros(m,1);

xx = x;
mm = m;
idx_sig = (1:m)'; %location of significant x
ind = 1;
while ind == 1
    tmp = (xx'*xx)\eye(mm);
    cest = tmp*(xx'*y);
    u = y - xx*cest;
    uvar = u'*u/n;
    if ind_const_var == 1 %OLS std error
        cvarmat = uvar*tmp;
    else %White std error
        tmp_mid = xx'*diag(u.^2)*xx;
        cvarmat = tmp*tmp_mid*tmp;
    end
    cstd = sqrt(diag(cvarmat));
    czscore = cest./cstd;
    cpvalue = 2*(1-normcdf(abs(czscore)));

    if mm == 1 %terminate if only the intercept is left
        ind = 0;
    else
        if max(cpvalue(2:mm)) <= pvalue_star %terminate if all pvalue of nonconstant x are small
            ind = 0;
        else
            [~,idx_tmp] = max(cpvalue(2:mm));
            idx = idx_tmp + 1; %take into account the intercept
            idx_sig = setdiff(idx_sig, idx_sig(idx));
            xx = x(:,idx_sig); %eliminate the least significant nonconstant x
            mm = size(xx,2);
        end
    end

end
est.best(idx_sig) = cest;
est.bstd(idx_sig) = cstd;
est.bzscore(idx_sig) = czscore;
est.bpvalue(idx_sig) = cpvalue;
est.ind_sig(idx_sig) = 1;


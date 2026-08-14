clear;
dbstop if warning;
dbstop if error;
rng(123456789);
%addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Feb\Functions'));

%% Simulated data
ind_SV = 0; %if SV is present

n = 100*10;
K = 6;
disp(['n = ', num2str(n), ', K = ', num2str(K)]);
x = [ones(n,1)  randn(n, K-1)];
beta_true = [1; 1; zeros(K-2,1)];

sig2_mu = 0.1;
sig2_rho = 0.9;
sig2_s = 0.1;
sig2_true = sig2_mu * ones(n,1);

y = zeros(n,1);
for t = 1:n
    ut = sqrt(sig2_true(t)) * randn; 
    y(t) = x(t,:) * beta_true + ut;
    
    if ind_SV == 1
        if t < n
            logsig2_lag = log(sig2_true(t));
            logsig2 = (1-sig2_rho) * sig2_mu + sig2_rho * logsig2_lag + sqrt(sig2_s) * randn;
            sig2_true(t+1) = exp(logsig2);
        end
    end
end


%% MCMC
burnin = 100;
ndraws = 1000;
disp(['burnin = ', num2str(burnin), ', ndraws = ', num2str(ndraws)]);
var_b0 = 100;

% draws = Sampler_LinearReg_HS(y, x, burnin, ndraws, var_b0, ind_SV);
% draws = Sampler_LinearReg_SpikeSlab(y, x, burnin, ndraws, ind_SV);

pvalue_star = 0.05;
ind_const_var = 1;
[best,bstd,bzscore,bpvalue,ind_sig] = Stepwise_Regression(y,x,pvalue_star,ind_const_var);




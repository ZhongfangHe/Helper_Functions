% Currently, intercept is shrunk as well. 
% May need to change this.


% Estimate a linear regression with spike-slab shrinkage for nonconstant regressors
% yt = b0 + xt b1' + N(0,st), t=1,...,n
% b0 ~ N(0,100)
% b1j ~ N(0, indj * sj + (1-indj) * c * sj)
% p(indj = 1) = q, p(sj) = IG(s_a0, s_b0), c~=0 (e.g. 1e-6), p(q)=U(0,1).
% st could be SV or constant with Jeffery's prior

function draws = Sampler_LinearReg_SpikeSlab(y, x, burnin, ndraws, ind_SV)
% Inputs:
%   y: a n-by-1 vector of target data.
%   x: a n-by-K matrix of regressor data (including constant).
%   burnin: a scalar of the number of burnins.
%   ndraws: a scalar of the number of effective draws.
%   var_beta0: a scalar of prior variance of the intercept.
%   ind_SV: an indicator if SV for noise variance.
% Outputs:
%   draws: a structure of the final draws.
%     if ind_SV = 0, "draws" includes the following fields:
%       yfit: a ndraws-by-n matrix of fitted values of yt. 
%       beta: a ndraws-by-K matrix of coef.
%       taul: a ndraws-by-2 matrix of global var and its hyperpara.
%       phil: a ndraws-by-2K matrix of local var and its hyperpara.
%       sig2: a ndraws-by-1 vector of noise variance.
%     if ind_SV = 1, "draws" includes the following fields:
%       yfit: a ndraws-by-n matrix of fitted values of yt. 
%       beta: a ndraws-by-K matrix of coef.
%       taul: a ndraws-by-2 matrix of global var and its hyperpara.
%       phil: a ndraws-by-2K matrix of local var and its hyperpara.
%       sig2: a ndraws-by-n matrix of noise variance.
%       SVpara: a ndraws-by-6 matrix of SV parameters.


[n,K] = size(x);



%% Priors: beta, beta0 ~ N(0,100), beta1 ~ N(0, taul * diag(phil)), taul, phil are IBs
c = 1e-6;

s_a = 5;
s_b = 5;%4;
s_prior = [s_a  s_b]';

beta = zeros(K,1);
ind = zeros(K,1);


%% Priors: SV or constant measurement noise variance
if ind_SV == 1
% long-run mean: p(mu) ~ N(mu0, Vmu), e.g. mu0 = 0; Vmu = 10;
% persistence: p(phi) ~ N(phi0, Vphi)I(-1,1), e.g. phi0 = 0.95; invVphi = 0.04;
% variance: p(sig2) ~ G(0.5, 2*sig2_s), sig2_s ~ IG(0.5,1/lambda), lambda ~ IG(0.5,1)    
    muh0 = 0; invVmuh = 1/10; % mean: p(mu) ~ N(mu0, Vmu)
    phih0 = 0.95; invVphih = 1/0.04; % AR(1): p(phi) ~ N(phi0, Vphi)I(-1,1)
    priorSV = [muh0 invVmuh phih0 invVphih]'; %collect prior hyperparameters
    muh = muh0 + sqrt(1/invVmuh) * randn;
    phih = phih0 + sqrt(1/invVphih) * trandn((-1-phih0)*sqrt(invVphih),(1-phih0)*sqrt(invVphih));

    lambdah = 1/gamrnd(0.5,1);
    sigh2_s = 1/gamrnd(0.5,lambdah);
    sigh2 = gamrnd(0.5,2*sigh2_s);
    sigh = sqrt(sigh2);

    hSV = log(var(y))*ones(n,1); %initialize by log OLS residual variance.
    vary = exp(hSV);
else %Jeffery's prior p(sig2) \prop 1/sig2
    sig2 = var(y); %initialize
    vary = sig2 * ones(n,1);
end


%% MCMC
draws.beta = zeros(ndraws,K);
draws.ind = zeros(ndraws,K);
draws.s = zeros(ndraws,K);
draws.q = zeros(ndraws,1);
    
if ind_SV == 1
    draws.SVpara = zeros(ndraws,6); % [mu phi sig2 sig sig2_s lambda]
    draws.sig2 = zeros(ndraws,n); %residual variance
else
    draws.sig2 = zeros(ndraws,1);
end

draws.yfit = zeros(ndraws,n);

tic;
ntotal = burnin + ndraws;
for drawi = 1:ntotal 
    % Update beta
    [beta, ind, s, q] = SpikeSlab_LinearReg_marg_single(y, x, vary, beta, ...
            ind, c, s_prior); %marginalize out beta, single move
    
    
    % Residual variance
    yfit = x * beta;
    eps = y - yfit;
    if ind_SV == 1
        logz2 = log(eps.^2 + 1e-100);
        [hSV, muh, phih, sigh, sigh2_s, lambdah] = SV_update_asis(logz2, hSV, ...
            muh, phih, sigh, sigh2_s, lambdah, priorSV);    
        vary = exp(hSV);  
%         vary(isinf(vary)) = maxNum;
    else
        sig2 = 1/gamrnd(0.5*n, 2/(eps'*eps));
        vary = sig2 * ones(n,1); 
    end      
   

    % Collect draws
    if drawi > burnin
        i = drawi - burnin;
        
        if ind_SV == 1
            draws.sig2(i,:) = vary';
            draws.SVpara(i,:) = [muh phih sigh^2 sigh sigh2_s lambdah];
        else
            draws.sig2(i) = sig2;
        end
        
        draws.beta(i,:) = beta';
        draws.ind(i,:) = ind';
        draws.s(i,:) = s';
        draws.q(i) = q;
        
        draws.yfit(i,:) = yfit';
    end
    
    
    % Display elapsed time
    if (drawi/5000) == round(drawi/5000)
        disp([num2str(drawi), ' out of ', num2str(ntotal),' draws have completed!']);
        toc;
    end    
end    


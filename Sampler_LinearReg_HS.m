% Estimate a linear regression with horseshoe shrinkage for nonconstant regressors
% yt = b0 + xt b1' + N(0,st), t=1,...,n
% b0 ~ N(0,100)
% b1j ~ N(0,tau * tauj), sqrt(tau)~C+(0,1/n), tauj are IB(0.5,0.5)
% st could be SV or constant with Jeffery's prior

function draws = Sampler_LinearReg_HS(y, x, burnin, ndraws, var_beta0, ind_SV)
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
phil_d = 1./gamrnd(0.5,1,K-1,1);
phil = 1./gamrnd(0.5*ones(K-1,1),phil_d); %local variances
taul_d = 1/gamrnd(0.5,1);
taul = 1/gamrnd(0.5, taul_d); %global variance
%var_beta0 = 100;
psil = [var_beta0; taul*phil]; 
% beta = sqrt(psil) .* randn(K,1);


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
draws.taul = zeros(ndraws,2);
draws.phil = zeros(ndraws,2*(K-1));
draws.beta = zeros(ndraws,K); 
    
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
    sigy = sqrt(vary);
    ystar = y./sigy;
    xstar = x./repmat(sigy,1,K);
    A_inv = diag(1./psil) + xstar' * xstar;   
    Ainva = xstar'*ystar;
    [U,D,~] = svd(A_inv);
    Ddiag = diag(D);
    if min(Ddiag) < realmin
        error("Min singular value < realmin");
    else
        Dinvdiag = 1./Ddiag;
        Dinv = diag(Dinvdiag);
        tmp = Dinv*U'*Ainva + sqrt(Dinvdiag).*randn(K,1);
        beta = U*tmp;
    end
%     if rcond(A_inv) > 1e-15
%         A_inv_half = chol(A_inv);
%         a = A_inv \ (xstar' * ystar);
%         beta = a + A_inv_half \ randn(K,1);
%     else
%         A = robust_inv(A_inv);
%         A_half = robust_chol(A);
%         a = A * (xstar' * ystar);
%         beta = a + A_half * randn(K,1);
%     end        
    
    
    % Hyperparameters of beta
    beta2 = beta(2:K).^2;
%     [taul, taul_d, phil, phil_d] = Horseshoe_update_vector(beta2,...
%         taul, taul_d, phil, phil_d); 
    [taul, taul_d, phil, phil_d] = Horseshoe_update_vector_scaled(beta2,...
        taul, taul_d, phil, phil_d, 1/n);     
    psil = [var_beta0; taul*phil];
    
    
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
        draws.taul(i,:) = [taul  taul_d];
        draws.phil(i,:) = [phil'  phil_d'];
        
        draws.yfit(i,:) = yfit';
    end
    
    
    % Display elapsed time
    if (drawi/5000) == round(drawi/5000)
        disp([num2str(drawi), ' out of ', num2str(ntotal),' draws have completed!']);
        toc;
    end    
end    
    
    
    
    


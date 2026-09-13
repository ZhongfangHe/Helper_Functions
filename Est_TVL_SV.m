% Estimate the TVL model:
% yt = xt'*alpha + sigma*f(zt) + ut, ut~N(0,st), f~N(0,K) with K(i,j)=exp(-(phi^2)*|zi-zj|^2)).
% st follows SV.


function draws = Est_TVL_SV(y,x,z,burnin,ndraws,alpha0_var)
% Inputs:
%   y: a n-by-1 vector of targets.
%   x: a n-by-m matrix of linear regressors.
%   z: a n-by-mz matrix of nonlinear regressors. 
%   burnin: an integer of the number of burn-ins.
%   ndraws: an integer of the number of draws after burn-in.
%   alpha0_var: a scalar of the prior variance of the intercept (e.g. 100)
% Outputs:
%   draws: a structure with the following fields.
%     draws.alpha: a ndraws-by-m matrix of linear coef.
%     draws.sigma: a ndraws-by-1 vector of sigma.
%     draws.phi: a ndraws-by-1 vector of phi.
%     draws.f: a ndraws-by-n matrix of f.
%     draws.yfit: a ndraws-by-n matrix of xt'*alpha+sigma*f(zt).
%     draws.tau: a ndraws-by-1 vector of global para of [alpha(2:m);sigma].
%     draws.tau0: a ndraws-by-1 vector of hyperpara for tau.
%     draws.lambda: a ndraws-by-m matrix of local para of [alpha(2:m);sigma].
%     draws.lambda0: a ndraws-by-m matrix of hyperpara for lambda.
%     draws.phi_lambda: a ndraws-by-1 vector of local para of phi.
%     draws.phi_lambda0: a ndraws-by-1 vector of hyperpara for phi_lambda.
%     draws.rw: a ndraws-by-2 matrix of MH tuning para for [phi sigma].
%     draws.ff: a ndraws-by-n matrix of sigma*f.
%     draws.xalpha: a ndraws-by-n matrix of x*alpha.
%     draws.count_phi: a scalar of MH acceptance rate for phi.
%     draws.count_sigma: a scalar of MH acceptance rate for sigma.
%     draws.s: a ndraws-by-n matrix of residual variance st.
%     draws.SVpara: a ndraws-by-6 matrix of SV parameters.


ntotal = burnin + ndraws;
[n,m] = size(x);

%alpha0 = sqrt(alpha0_var)*randn; %prior draw of intercept from N(0,alpha0_var)

n2 = n^2;
tau0 = 1/gamrnd(0.5,1/n2);
tau = 1/gamrnd(0.5,tau0);
lambda0 = 1./gamrnd(0.5,1,m,1);
lambda = 1./gamrnd(0.5,lambda0);
beta = sqrt(tau)*sqrt(lambda).*randn(m,1); %prior draw of alpha(2:m) and sigma
sigma = beta(m);
sigma2 = sigma^2;
%alpha = [alpha0; beta(1:m-1)]; 

phi_lambda0 = 1/gamrnd(0.5,1);
phi_lambda = 1/gamrnd(0.5,phi_lambda0);
phi = sqrt(phi_lambda)*randn; %prior draw of phi

Ktmp = kernel_cov_matrix(z);
Kcov = Ktmp.^(phi^2);
%[Ku,Kd,~] = svd(Kcov);
%Kd_diag = diag(Kd);
%ftmp = sqrt(Kd_diag).*randn(n,1);
%f = Ku*ftmp; %prior draw of nonlinear part f

%% Priors: SV or constant measurement noise variance
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

KK = sigma2*Kcov + diag(vary);
[KKu,KKd,~] = svd(KK);
KKd_diag = diag(KKd);


pstar = 0.44; %target acceptance prob of MH
rw_phi = 0.01;
rw_sigma = 0.01; %stdev of MH steps 
count_phi = 0; 
count_sigma = 0; %MH acceptance counter

draws.alpha = zeros(ndraws,m);
draws.sigma = zeros(ndraws,1);
draws.phi = zeros(ndraws,1);
draws.f = zeros(ndraws,n);
draws.yfit = zeros(ndraws,n);
draws.tau = zeros(ndraws,1);
draws.tau0 = zeros(ndraws,1);
draws.lambda = zeros(ndraws,m);
draws.lambda0 = zeros(ndraws,m);
draws.phi_lambda = zeros(ndraws,1);
draws.phi_lambda0 = zeros(ndraws,1);
draws.rw = zeros(ndraws,2); %phi,sigma
draws.ff = zeros(ndraws,n); %sigma*f
draws.xalpha = zeros(ndraws,n); %x*alpha
draws.count_phi = 0;
draws.count_sigma = 0;
draws.SVpara = zeros(ndraws,6); % [mu phi sig2 sig sig2_s lambda]
draws.s = zeros(ndraws,n); %residual variance
tic;
for drawi = 1:ntotal    
    % Draw alpha (integrate out f)
    tmpu = KKu;
    tmpd_diag = KKd_diag;
    tmp_ux = tmpu'*x;
    tmp_uy = tmpu'*y;
    tmpd_inv = diag(1./tmpd_diag);
    Binv = diag([1/alpha0_var; 1./(tau*lambda(1:m-1))]) + tmp_ux'*tmpd_inv*tmp_ux;
    Binvb = tmp_ux'*tmpd_inv*tmp_uy;
    [BinvU,BinvD,~] = svd(Binv);
    BinvD_diag = diag(BinvD);
    ub = diag(1./BinvD_diag)*BinvU'*Binvb;
    alpha_tmp = ub + sqrt(1./BinvD_diag).*randn(m,1);
    alpha = BinvU*alpha_tmp;


    % Draw sigma (integrate out f)
    sigma_old = sigma;
    sigma_new = sigma + rw_sigma*randn;

    sigma2_old = sigma_old^2;
    tmp = y-x*alpha;
    KKu_old = KKu;
    KKd_diag_old = KKd_diag;
    tmpd_diag = KKd_diag_old;
    logdet_tmp1 = sum(log(tmpd_diag));
    tmpu = KKu_old;
    tmp2 = tmpu'*tmp;
    tmp3 = tmp2'*diag(1./tmpd_diag)*tmp2;
    loglike_old = -0.5*logdet_tmp1-0.5*tmp3;
    logprior_old = -0.5*sigma2_old/(tau*lambda(m));

    sigma2_new = sigma_new^2;
    KK_new = sigma2_new*Kcov + diag(vary);
    [KKu_new,KKd_new,~] = svd(KK_new);
    KKd_diag_new = diag(KKd_new);
    tmpd_diag = KKd_diag_new;
    logdet_tmp1 = sum(log(tmpd_diag));
    tmpu = KKu_new;
    tmp2 = tmpu'*tmp;
    tmp3 = tmp2'*diag(1./tmpd_diag)*tmp2;
    loglike_new = -0.5*logdet_tmp1-0.5*tmp3;
    logprior_new = -0.5*sigma2_new/(tau*lambda(m)); 

    log_accpt_prob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand) <= log_accpt_prob
        sigma = sigma_new;
        KKu = KKu_new;
        KKd_diag = KKd_diag_new;
        if drawi > burnin
            count_sigma = count_sigma + 1;
        end
    else
        sigma = sigma_old;
        KKu = KKu_old;
        KKd_diag = KKd_diag_old;        
    end
    sigma2 = sigma^2;

    accpt_prob = min(1,exp(log_accpt_prob));
    logrw_new = log(rw_sigma) + (accpt_prob - pstar)/(drawi*pstar*(1-pstar));
    rw_sigma = exp(logrw_new);


    % Draw phi (integrate out f)
    phi_old = phi;
    phi_new = phi + rw_phi*randn;

    Kcov_old = Kcov;
    KKu_old = KKu;
    KKd_diag_old = KKd_diag;
    tmpd_diag = KKd_diag_old;
    logdet_tmp1 = sum(log(tmpd_diag));
    tmpu = KKu_old;
    tmp2 = tmpu'*tmp;
    tmp3 = tmp2'*diag(1./tmpd_diag)*tmp2;
    loglike_old = -0.5*logdet_tmp1-0.5*tmp3;
    logprior_old = -0.5*phi_old*phi_old/phi_lambda;

    Kcov_new = Ktmp.^(phi_new^2);
    KK_new = sigma2*Kcov_new + diag(vary);
    [KKu_new,KKd_new,~] = svd(KK_new);
    KKd_diag_new = diag(KKd_new);
    tmpd_diag = KKd_diag_new;
    logdet_tmp1 = sum(log(tmpd_diag));
    tmpu = KKu_new;
    tmp2 = tmpu'*tmp;
    tmp3 = tmp2'*diag(1./tmpd_diag)*tmp2;
    loglike_new = -0.5*logdet_tmp1-0.5*tmp3;
    logprior_new = -0.5*phi_new*phi_new/phi_lambda;    

    log_accpt_prob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand) <= log_accpt_prob
        phi = phi_new;
        Kcov = Kcov_new;
        KKu = KKu_new;
        KKd_diag = KKd_diag_new;
        if drawi > burnin
            count_phi = count_phi + 1;
        end
    else
        phi = phi_old;
        Kcov = Kcov_old;
        KKu = KKu_old;
        KKd_diag = KKd_diag_old;        
    end

    accpt_prob = min(1,exp(log_accpt_prob));
    logrw_new = log(rw_phi) + (accpt_prob - pstar)/(drawi*pstar*(1-pstar));
    rw_phi = exp(logrw_new);



    % Draw f
    fB = (1/sigma2) * diag(vary) - (1/sigma2) * diag(vary) * (KKu*diag(1./KKd_diag)*KKu') * diag(vary);
    [fBV, fBE, ~] = svd(fB);
    Binvb = sigma*diag(1./vary)*(y-x*alpha);
    ftmp = fBE*fBV'*Binvb + sqrt(diag(fBE)).*randn(n,1);
    f = fBV*ftmp;


    % ASIS step for sigma, f
    [Ku,Kd,~] = svd(Kcov);
    Kd_diag = diag(Kd);
    ff = sigma*Ku'*f;
    sigma_sign = sign(sigma);

    sigma2_p = 0.5-0.5*n;
    sigma2_a = 1/(tau*lambda(m));
    sigma2_b = ff'*diag(1./Kd_diag)*ff;
    sigma2_asis = gigrnd(sigma2_p, sigma2_a, sigma2_b, 1);
    sigma = sqrt(sigma2_asis)*sigma_sign;

    f = Ku*ff/sigma;


    % Draw tau, tau0
    beta = [alpha(2:m); sigma];
    beta2 = beta.^2;
    tau_a = (1+m)/2;
    tau_b = 1/tau0 + 0.5*sum(beta2./lambda);
    tau = 1/gamrnd(tau_a,1/tau_b);

    tau0_a = 1;
    tau0_b = n2 + 1/tau;
    tau0 = 1/gamrnd(tau0_a, 1/tau0_b);


    % Draw lambda, lambda0
    lambda_a = 1;
    lambda_b = 1./lambda0 + 0.5*beta2/tau;
    lambda = 1./gamrnd(lambda_a, 1./lambda_b);

    lambda0_a = 1;
    lambda0_b = 1+1./lambda;
    lambda0 = 1./gamrnd(lambda0_a, 1./lambda0_b);


    % Draw phi_lambda, phi_lambda0
    phi_lambda = 1/gamrnd(1,1/(1/phi_lambda0+ 0.5*phi*phi));
    phi_lambda0 = 1/gamrnd(1,1/(1+1/phi_lambda));


    % Residual variance
    yfit = x*alpha+sigma*f;
    eps = y - yfit;
    logz2 = log(eps.^2 + 1e-100);
    [hSV, muh, phih, sigh, sigh2_s, lambdah] = SV_update_asis(logz2, hSV, ...
        muh, phih, sigh, sigh2_s, lambdah, priorSV);    
    vary = exp(hSV);  
%         vary(isinf(vary)) = maxNum;
    KK = sigma2*Kcov + diag(vary);
    [KKu,KKd,~] = svd(KK);
    KKd_diag = diag(KKd);



    % Collect draws
    if drawi > burnin
        draws.alpha(drawi-burnin,:) = alpha';
        draws.sigma(drawi-burnin) = sigma;
        draws.tau(drawi-burnin) = tau;
        draws.tau0(drawi-burnin) = tau0;
        draws.lambda(drawi-burnin,:) = lambda';
        draws.lambda0(drawi-burnin,:) = lambda0;
        draws.phi_lambda(drawi-burnin) = phi_lambda';
        draws.phi_lambda0(drawi-burnin) = phi_lambda0;
        draws.s(drawi-burnin,:) = vary';
        draws.SVpara(drawi-burnin,:) = [muh phih sigh^2 sigh sigh2_s lambdah];
        draws.phi(drawi-burnin) = phi;
        draws.f(drawi-burnin,:) = f';
        draws.yfit(drawi-burnin,:) = (x*alpha+sigma*f)';
        draws.ff(drawi-burnin,:) = (sigma*f)';
        draws.xalpha(drawi-burnin,:) = (x*alpha)';
        draws.rw(drawi-burnin,:) = [rw_phi rw_sigma];
        draws.count_phi = count_phi/ndraws;
        draws.count_sigma = count_sigma/ndraws;
    end
    if round(drawi/1000) == (drawi/1000)
        disp([num2str(drawi),' draws have been completed!']);
        toc;
        disp(' ');
    end
end



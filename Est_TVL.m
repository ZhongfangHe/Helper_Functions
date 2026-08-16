% Estimate the TVL model:
% yt = xt'*alpha + sigma*f(zt) + ut, ut~N(0,s), f~N(0,K) with K(i,j)=exp(-(phi^2)*|zi-zj|^2)).


function draws = Est_TVL(y,x,z,burnin,ndraws,alpha0_var)
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
%     draws.s: a ndraws-by-1 vector of residual variance s.
%     draws.phi: a ndraws-by-1 vector of phi.
%     draws.f: a ndraws-by-n matrix of f.
%     draws.yfit: a ndraws-by-n matrix of xt'*alpha+sigma*f(zt).
%     draws.tau: a ndraws-by-1 vector of global para of [alpha(2:m);sigma].
%     draws.tau0: a ndraws-by-1 vector of hyperpara for tau.
%     draws.lambda: a ndraws-by-m matrix of local para of [alpha(2:m);sigma].
%     draws.lambda0: a ndraws-by-m matrix of hyperpara for lambda.
%     draws.phi_lambda: a ndraws-by-1 vector of local para of phi.
%     draws.phi_lambda0: a ndraws-by-1 vector of hyperpara for phi_lambda.
%     draws.s0: a ndraws-by-1 vector of local para of s.
%     draws.rw: a ndraws-by-2 matrix of MH tuning para for [phi sigma].
%     draws.ff: a ndraws-by-n matrix of sigma*f.
%     draws.xalpha: a ndraws-by-n matrix of x*alpha.
%     draws.count_phi: a scalar of MH acceptance rate for phi.
%     draws.count_sigma: a scalar of MH acceptance rate for sigma.


ntotal = burnin + ndraws;
[n,m] = size(x);

alpha0 = sqrt(alpha0_var)*randn; %prior draw of intercept from N(0,alpha0_var)

n2 = n^2;
tau0 = 1/gamrnd(0.5,1/n2);
tau = 1/gamrnd(0.5,tau0);
lambda0 = 1./gamrnd(0.5,1,m,1);
lambda = 1./gamrnd(0.5,lambda0);
beta = sqrt(tau)*sqrt(lambda).*randn(m,1); %prior draw of alpha(2:m) and sigma
sigma = beta(m);
alpha = [alpha0; beta(1:m-1)]; 

phi_lambda0 = 1/gamrnd(0.5,1);
phi_lambda = 1/gamrnd(0.5,phi_lambda0);
phi = sqrt(phi_lambda)*randn; %prior draw of phi

Ktmp = kernel_cov_matrix(z);
Kcov = Ktmp.^(phi^2);
[Ku,Kd,~] = svd(Kcov);
Kd_diag = diag(Kd);
ftmp = sqrt(Kd_diag).*randn(n,1);
f = Ku*ftmp; %prior draw of nonlinear part f

s0 = 1/gamrnd(0.5,1);
s_a0  = 0.5;
s_b0 = 1/s0;
% s = 1/gamrnd(0.5,s0); %prior draw of residual variance s~IG(0.5,1/s0),s0~IG(0.5,1)

pstar = 0.44; %target acceptance prob of MH
rw_phi = 0.01;
rw_sigma = 0.01; %stdev of MH steps 
count_phi = 0; 
count_sigma = 0; %MH acceptance counter

draws.alpha = zeros(ndraws,m);
draws.sigma = zeros(ndraws,1);
draws.s = zeros(ndraws,1);
draws.phi = zeros(ndraws,1);
draws.f = zeros(ndraws,n);
draws.yfit = zeros(ndraws,n);
draws.tau = zeros(ndraws,1);
draws.tau0 = zeros(ndraws,1);
draws.lambda = zeros(ndraws,m);
draws.lambda0 = zeros(ndraws,m);
draws.phi_lambda = zeros(ndraws,1);
draws.phi_lambda0 = zeros(ndraws,1);
draws.s0 = zeros(ndraws,1);
draws.rw = zeros(ndraws,2); %phi,sigma
draws.ff = zeros(ndraws,n); %sigma*f
draws.xalpha = zeros(ndraws,n); %x*alpha
draws.count_phi = 0;
draws.count_sigma = 0;

tic;
for drawi = 1:ntotal
    % Draw s
    s01 = s_a0 + 0.5*n;
    xx = [x f];
    eps = y - xx*[alpha; sigma];
    s02 = s_b0 + 0.5*(eps'*eps);
    s = 1/gamrnd(s01,1/s02);


    % Draw s0
    s0 = 1/gamrnd(1,1/(1+1/s));
    s_b0 = 1/s0;

    
    % Draw alpha (integrate out f)
    sigma2 = sigma^2;
    tmpu = Ku;
    tmpd_diag = s + sigma2*Kd_diag;
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
    tmpu = Ku;
    tmpd_diag = s+sigma2_old*Kd_diag;
    logdet_tmp1 = sum(log(tmpd_diag));
    tmp2 = tmpu'*tmp;
    tmp3 = tmp2'*diag(1./tmpd_diag)*tmp2;
    loglike_old = -0.5*logdet_tmp1-0.5*tmp3;
    logprior_old = -0.5*sigma2_old/(tau*lambda(m));

    sigma2_new = sigma_new^2;
    tmpd_diag = s+sigma2_new*Kd_diag;
    logdet_tmp1 = sum(log(tmpd_diag));
    tmp3 = tmp2'*diag(1./tmpd_diag)*tmp2;
    loglike_new = -0.5*logdet_tmp1-0.5*tmp3;
    logprior_new = -0.5*sigma2_new/(tau*lambda(m)); 

    log_accpt_prob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand) <= log_accpt_prob
        sigma = sigma_new;
        if drawi > burnin
            count_sigma = count_sigma + 1;
        end
    else
        sigma = sigma_old;
    end
    sigma2 = sigma^2;

    accpt_prob = min(1,exp(log_accpt_prob));
    logrw_new = log(rw_sigma) + (accpt_prob - pstar)/(drawi*pstar*(1-pstar));
    rw_sigma = exp(logrw_new);


    % Draw phi (integrate out f)
    phi_old = phi;
    phi_new = phi + rw_phi*randn;

    Kcov_old = Ktmp.^(phi_old^2);
    [Ku_old,Kd_old,~] = svd(Kcov_old);
    Kd_old_diag = diag(Kd_old);
    tmpu = Ku_old;
    tmpd_diag = s + sigma2*Kd_old_diag;
    logdet_tmp1 = sum(log(tmpd_diag));
    tmp2 = tmpu'*tmp;
    tmp3 = tmp2'*diag(1./tmpd_diag)*tmp2;
    loglike_old = -0.5*logdet_tmp1-0.5*tmp3;
    logprior_old = -0.5*phi_old*phi_old/phi_lambda;

    Kcov_new = Ktmp.^(phi_new^2);
    [Ku_new,Kd_new,~] = svd(Kcov_new);
    Kd_new_diag = diag(Kd_new);
    tmpu = Ku_new;
    tmpd_diag = s + sigma2*Kd_new_diag;
    logdet_tmp1 = sum(log(tmpd_diag));
    tmp2 = tmpu'*tmp;
    tmp3 = tmp2'*diag(1./tmpd_diag)*tmp2;
    loglike_new = -0.5*logdet_tmp1-0.5*tmp3;
    logprior_new = -0.5*phi_new*phi_new/phi_lambda;    

    log_accpt_prob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand) <= log_accpt_prob
        phi = phi_new;
        Ku = Ku_new;
        Kd_diag = Kd_new_diag;
        if drawi > burnin
            count_phi = count_phi + 1;
        end
    else
        phi = phi_old;
        Ku = Ku_old;
        Kd_diag = Kd_old_diag;        
    end

    accpt_prob = min(1,exp(log_accpt_prob));
    logrw_new = log(rw_phi) + (accpt_prob - pstar)/(drawi*pstar*(1-pstar));
    rw_phi = exp(logrw_new);



    % Draw f
    sigma_s = sigma/s;
    sigma2_s = sigma2/s;
    tmp_diag = Kd_diag./(1+sigma2_s*Kd_diag);
    Ku_times_b = sigma_s*diag(tmp_diag)*Ku'*(y-x*alpha);
    ftmp = Ku_times_b + sqrt(tmp_diag).*randn(n,1);
    f = Ku*ftmp;


    % ASIS step for sigma, f
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
        draws.s0(drawi-burnin) = s0;
        draws.s(drawi-burnin) = s;
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
































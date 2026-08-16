clear;
dbstop if warning;
dbstop if error;
rng(123456789);
%addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Feb\Functions'));


%% Simulated data (DGP = ARX+possible GP)
ind_SV = 0; %if SV is present

n = 100;
K = 2;
disp(['n = ', num2str(n), ', K = ', num2str(K)]);
xex = randn(n, K);
beta_true = [1; zeros(K-1,1)];

sig2_mu = 0.1;
sig2_rho = 0.9;
sig2_s = 0.1;
sig2_true = sig2_mu * ones(n,1);

idx_f = 1; %if nonlinear part is present
switch idx_f
    case 1 %from kernel
        Kcov_true = kernel_cov_matrix(xex(:,2:K));
        f_true = mvnrnd(zeros(n,1),Kcov_true)';
        disp('dgp = nonlinear:kernel');
    case 2 %from TVP
        f_true = -0.5*(log(abs(xex(:,1))) - mean(log(abs(xex(:,1)))));
        disp('dgp = nonlinear:TVP');
    case 3 %from linear
        f_true = zeros(n,1);
        disp('dgp = linear');
end
disp(' ');

y = zeros(n,1);
for t = 1:n
    ut = sqrt(sig2_true(t)) * randn;
    if t >= 3
        y(t) = 0.1 + 0.9*y(t-2) + xex(t,:) * beta_true + f_true(t) + ut;
    else
        y(t) = 0.1 + xex(t,:) * beta_true + f_true(t) + ut;
    end

    if ind_SV == 1
        if t < n
            logsig2_lag = log(sig2_true(t));
            logsig2 = (1-sig2_rho) * sig2_mu + sig2_rho * logsig2_lag + sqrt(sig2_s) * randn;
            sig2_true(t+1) = exp(logsig2);
        end
    end
end



%% Split estimation and forecast samples
nest = n-12; %round(0.8*n);
npred = n - nest;



%% Assemble target and regressors for estimation
nlag = 3; %max number of lags
yest = y(nlag+1:nest);
xest = [ones(nest-nlag,1) zeros(nest-nlag,nlag+K)];
for j = 1:nlag
    xest(:,j+1) = y(nlag+1-j:nest-j);
end
xest(:,nlag+2:nlag+K+1) = xex(nlag+1:nest,:);



%% Estimate and forecast
model_list = {'OLS','HS','TVL'};
nmodel = length(model_list);
model_est = cell(nmodel);
ypred_model = zeros(npred,nmodel);
for idx_model = 1:nmodel 
    switch idx_model
        case 1  %OLS
            % Estimate
            pvalue_star = 0.05;
            ind_const_var = 1;
            est = Est_LinearReg_Stepwise(yest,xest,pvalue_star,ind_const_var);
            
            % Predict
            ylag_pred = y(nest-nlag+1:nest); %K-by-1 vecor [y_{t-K+1}; y_{t-K+2}; ... ; y_t]
            xpred = xex(nest+1:nest+npred,:); %H-by-m matrix stacking [x_{t+1}'; x_{t+2}'; ... ; x_{t+H}']
            theta = est.best; %(1+K+m)-by-1 vector of linear parameters [a,b1,...,bk,c].
            ypred_tmp = Pred_ARX_mean(ylag_pred,xpred,theta); 

            % Collect results
            model_est{idx_model} = est;
            ypred_model(:,idx_model) = ypred_tmp;

    
        case 2 %Bayesian Horseshoe
            % Estimate
            burnin = 100;
            ndraws = 1000;
            disp(['burnin = ', num2str(burnin), ', ndraws = ', num2str(ndraws)]);
            var_b0 = 100;
            draws = Est_LinearReg_HS(yest, xest, burnin, ndraws, var_b0, ind_SV);
            % draws = Est_LinearReg_SpikeSlab(yest, xest, burnin, ndraws, ind_SV);
    
            % Predict
            ylag_pred = y(nest-nlag+1:nest); %K-by-1 vecor [y_{t-K+1}; y_{t-K+2}; ... ; y_t]
            xpred = xex(nest+1:nest+npred,:); %H-by-m matrix stacking [x_{t+1}'; x_{t+2}'; ... ; x_{t+H}']
            ypred_tmp = 0;
            for drawi = 1:ndraws
                theta = draws.beta(drawi,:)'; %(1+K+m)-by-1 vector of linear parameters [a,b1,...,bk,c].
                ypred_drawi = Pred_ARX_mean(ylag_pred,xpred,theta);
                ypred_tmp = ypred_tmp + ypred_drawi/ndraws;
            end

            % Collect results
            model_est{idx_model} = draws;
            ypred_model(:,idx_model) = ypred_tmp; 


        case 3 %Bayesian TVL
            % Estimate
            burnin = 1000*1;
            ndraws = 3000;
            disp(['burnin = ', num2str(burnin), ', ndraws = ', num2str(ndraws)]);
            alpha0_var = 10;
            zest = xex(nlag+1:nest,:); %regressors of the nonlinear part  
            draws = Est_TVL(yest,xest,zest,burnin,ndraws,alpha0_var);
    
            % Predict
            ylag_pred = y(nest-nlag+1:nest); %K-by-1 vecor [y_{t-K+1}; y_{t-K+2}; ... ; y_t]
            xpred = xex(nest+1:nest+npred,:); %H-by-m matrix stacking [x_{t+1}'; x_{t+2}'; ... ; x_{t+H}']
            zpred = xpred; %regressors of the nonlinear part 
            Ktmp = kernel_cov_matrix([zest;zpred]); %base part of kernel covariance matrix
            ypred_tmp = 0;
            for drawi = 1:ndraws
                theta = draws.alpha(drawi,:)'; %(1+K+m)-by-1 vector of linear parameters [a,b1,...,bk,c]
                sigmai = draws.sigma(drawi); %scaling factor of kernel covariance matrix
                phii = draws.phi(drawi); %para of kernle covariance matrix
                si = draws.s(drawi); %residual variance
                KM = (sigmai^2) * (Ktmp.^(phii^2)); %(t-K+H)-by-(t-K+H) covariance matrix of the GP
                [KMU,KMD,~] = svd(KM(1:nest-nlag,1:nest-nlag));
                tmpvec = diag(KMD)./(diag(KMD) + si);
                tmp1 = KMU * diag(tmpvec) * KMU';
                tmp2 = yest - xest*theta;
                fpm = tmp1*tmp2; %(t-K)-by-1 vector of posterior mean E(f_{K+1},f_{K+2},...,f_t|It,para)
                ypred_drawi = Pred_ARXGP_mean(ylag_pred,xpred,theta,KM,fpm);
                ypred_tmp = ypred_tmp + ypred_drawi/ndraws;
            end

            % Collect results
            model_est{idx_model} = draws;
            ypred_model(:,idx_model) = ypred_tmp;            
    end
    disp(['Model ', num2str(idx_model), ' (', model_list{idx_model}, ') out of ', num2str(nmodel), ' models is completed!']);
    disp(' ');
end



%% Evaluate predictions
ypred_true = y(nest+1:nest+npred);
pred_error = repmat(ypred_true,1,nmodel) - ypred_model;
pred_mse = diag(pred_error'*pred_error)/npred;
for j = 1:nmodel
    disp(['MSE of ',model_list{j}, ' = ', num2str(pred_mse(j))]);
end







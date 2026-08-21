% Multi-horizon forecast from an ARX-GP model, conditional on model parameters.
% yt = a + b1*ytm1 + ... + bk*ytmk + xt'*c + f(zt) + ut, ut is zero-mean noise.
% f(zt) is a Gaussian process with the prior [f_est; f_pred] ~ N(0,[G1 G2';G2 G3];
% Compute E(ytph|It,xtp1,...,xtph,theta) for h=1,2,...,H.

function ypred = Pred_ARXGP_mean(ylag,xpred,theta,KM,resid_var,yy)
% Inputs:
%   ylag: a K-by-1 vector of [y_{t-K+1}; y_{t-K+2}; ... ; y_t].
%   xpred: a H-by-m matrix stacking [x_{t+1}'; x_{t+2}'; ... ; x_{t+H}'].
%   theta: a (1+K+m)-by-1 vector of model parameters [a,b1,...,bk,c].
%   KM: a (t-K+H)-by-(t-K+H) covariance matrix of the GP.
%   resid_var: a scalar of the variance of the residual ut.
%   yy: a (t-K)-by-1 vector of yt-a-b1*ytm1-...-bk*ytmk-xt'*c.
% Outputs:
%   ypred: a H-by-1 vector of E(ytph|It,xtp1,...,xtph,para) for h=1,2,...,H.

[H,m] = size(xpred);
K = length(ylag);
if length(theta) ~= (1+K+m)
    error('length(theta) is not right!');
end


%% Compute predictive mean of f_pred
t = length(yy)+K;
G2 = KM(t-K+1:t-K+H,1:t-K);
G3 = KM(1:t-K,1:t-K); %KM(t-K+1:t-K+H,t-K+1:t-K+H);
[G3U,G3D,~] = svd(G3);
G3Ddiag = diag(G3D);
% if min(G3Ddiag) < realmin%
%     error("Min singular value < realmin");
% else
%     %G3Dinvdiag = 1./G3Ddiag;
%     %G3Dinv = diag(G3Dinvdiag);
%     %G3inv = G3U * G3Dinv * G3U';
%     %f_pred_mean = G2*G3inv*fpm;
%     f_pred_mean = G2*G3U*diag(1./(resid_var+G3Ddiag))*G3U'*yy;
% end
f_pred_mean = G2*G3U*diag(1./(resid_var+G3Ddiag))*G3U'*yy;



%% Stack into matrix form
am = theta(1);
bm = theta(2:K+1);
cm = theta(K+2:K+1+m);
if K == 1
    a = am;
    B = bm;
    C = cm';
    e = 1;
    dpred = f_pred_mean'; %each column is dtph = [ftph; zeros(K-1,1)];
else
    a = [am; zeros(K-1,1)];
    B = [bm';[eye(K-1) zeros(K-1,1)]];
    C = [cm';zeros(K-1,m)];
    e = [1;zeros(K-1,1)];
    dpred = [f_pred_mean';zeros(K-1,H)]; %each column is dtph = [ftph; zeros(K-1,1)];
end


%% Iterate forward
ypred = zeros(H,1);
tmp1 = a;
tmp2 = B*flipud(ylag);
xtp1 = xpred(1,:)';
tmp3 = C*xtp1;
dtp1 = dpred(:,1);
tmp4 = dtp1;
ypred(1) = e'*(tmp1 + tmp2 + tmp3 + tmp4);
if H > 1
    Bhm1 = B;
    for h = 2:H
        Bh = Bhm1 * B;
        tmp1 = tmp1 + Bhm1*a; %intercept
        tmp2 = B*tmp2; %ylag
        xtph = xpred(h,:)';
        tmp3 = B*tmp3 + C*xtph; %x
        dtph = dpred(:,h);
        tmp4 = B*tmp4 + dtph; %f
        ypred(h) = e'*(tmp1 + tmp2 + tmp3 + tmp4);
        Bhm1 = Bh;
    end
end



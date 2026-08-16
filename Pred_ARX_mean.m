% Multi-horizon forecast from an ARX model, conditional on model parameters.
% yt = a + b1*ytm1 + ... + bk*ytmk + xt'*c + ut, ut is zero-mean noise.
% Compute E(ytph|It,xtp1,...,xtph,theta) for h=1,2,...,H.

function ypred = Pred_ARX_mean(ylag,xpred,theta)
% Inputs:
%   ylag: a K-by-1 vector of [y_{t-K+1}; y_{t-K+2}; ... ; y_t].
%   xpred: a H-by-m matrix stacking [x_{t+1}'; x_{t+2}'; ... ; x_{t+H}'].
%   theta: a (1+K+m)-by-1 vector of model parameters [a,b1,...,bk,c].
% Outputs:
%   ypred: a H-by-1 vector of E(ytph|It,xtp1,...,xtph,theta) for h=1,2,...,H.

[H,m] = size(xpred);
K = length(ylag);
if length(theta) ~= (1+K+m)
    error('length(theta) is not right!');
end


%% Stack into matrix form
am = theta(1);
bm = theta(2:K+1);
cm = theta(K+2:K+1+m);
if K == 1
    a = am;
    B = bm;
    C = cm';
    e = 1; 
else
    a = [am; zeros(K-1,1)];
    B = [bm';[eye(K-1) zeros(K-1,1)]];
    C = [cm';zeros(K-1,m)];
    e = [1;zeros(K-1,1)];
end


%% Iterate forward
ypred = zeros(H,1);
tmp1 = a;
tmp2 = B*flipud(ylag);
xtp1 = xpred(1,:)';
tmp3 = C*xtp1;
ypred(1) = e'*(tmp1 + tmp2 + tmp3);
if H > 1
    Bhm1 = B;
    for h = 2:H
        Bh = Bhm1 * B;
        tmp1 = tmp1 + Bhm1*a; %intercept
        tmp2 = B*tmp2; %ylag
        xtph = xpred(h,:)';
        tmp3 = B*tmp3 + C*xtph; %x
        ypred(h) = e'*(tmp1 + tmp2 + tmp3);
        Bhm1 = Bh;
    end
end




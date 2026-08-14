% Sparsify b by SAVS (signal adaptive variable selector)
%
% if b > 1/||x||^(2/3), c = b - 1/(b * b * ||x||^2)
% if b < -1/||x||^(2/3), c = b + 1/(b * b * ||x||^2)
% otherwise, c = 0;


function c = SAVS(b, xnorm2, inv_xnorm23)
% Inputs:
%   b: a scalar of original number
%   xnorm2: a scalar of ||x||^2 (equivalently x'*x)
%   inv_xorm23: a scalar of 1/||x||^(2/3)
% Outputs:
%   c: a scalar of sparsified number

if b >= inv_xnorm23
    c = b - 1/(b * b * xnorm2);
elseif b <= -inv_xnorm23
    c = b + 1/(b * b * xnorm2);
else
    c = 0;
end
    
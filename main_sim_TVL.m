
n = 100;
m = 3;
x = [ones(n,1) rand(n,m-1)];
dgp = 1;
switch dgp 
    case 1 %from kernel
        Kcov_true = kernel_cov_matrix(x(:,2:m));
        f_true = mvnrnd(zeros(n,1),Kcov_true)';
        disp('dgp = kernel');
    case 2 %from TVP
        f_true = -0.5*(log(abs(x(:,2))) - mean(log(abs(x(:,2)))));
        disp('dgp = nonlinear');
    case 3 %from linear
        f_true = zeros(n,1);
        disp('dgp = linear');
end
ymean_true = 0.5*x(:,1) + 1.5*x(:,2) + f_true;
y = ymean_true + 0.1*randn(n,1);
disp(['n = ', num2str(n), ', m = ', num2str(m)]);


burnin = 1000*1;
ndraws = 3000;
alpha0_var = 1;
z = x(:,2:m); %regressors of the nonlinear part
draws = Est_TVL(y,x,z,burnin,ndraws,alpha0_var);
%draws = Est_TVL2(y,x,z,burnin,ndraws);



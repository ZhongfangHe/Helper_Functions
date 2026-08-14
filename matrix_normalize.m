
function xx = matrix_normalize(x)
[n,k] = size(x);
xx = zeros(n,k);
for j = 1:k
    meanj = mean(x(:,j));
    stdj = std(x(:,j));
    xx(:,j) = (x(:,j) - meanj)/stdj;
end

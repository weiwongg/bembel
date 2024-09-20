function error = my_integral(q_lvl, p, d)
num_samples = 2^q_lvl;
m = 1000;
integral_val = 0;
pts = haltonset(m,'Skip',1e3);
sigma = (1:m).^-p;
for i = 1:num_samples
y = pts(i,:);
%y = rand(1,m);
integral_val = integral_val + sum(sigma .* y.^d);
end
integral_val = integral_val / num_samples;
real_val = sum(sigma./(d+1));
error = abs(integral_val - real_val) / abs(real_val);
end

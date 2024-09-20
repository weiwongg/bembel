clear all;
close all;
max_lvl = 16
errors = zeros(3,max_lvl);
d = 4;
pp = [1 1.25 1.5 1.75]
for p = 1:4
for q_lvl = 1:max_lvl
    errors(p,q_lvl) = my_integral(q_lvl, pp(p), d);
end
end
errors = abs(errors)
semilogy(1:max_lvl,sqrt(2.^-(1:max_lvl)), '--')
hold on;
semilogy(1:max_lvl,errors(1,:))
hold on;
semilogy(1:max_lvl,errors(2,:))
hold on;
semilogy(1:max_lvl,errors(3,:))
hold on;
semilogy(1:max_lvl,errors(4,:))
legend('reference','p=1','p=1.25','p=1.5','p=1.75')
xlabel('level')
ylabel('error')
title('Convergence test of QMC')
data = zeros(max_lvl,5);
data(:,1) = 1:max_lvl
data(:,2:5) = errors
data(:,2:5) = errors'
dlmwrite('conv_test.txt',data,'delimiter','\t')
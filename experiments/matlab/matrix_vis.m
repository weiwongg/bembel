M = readmatrix('K.txt');
M = sparse(M(:,1),M(:,2),M(:,3));
d = eigs(M,200)
val = zeros(200,2);
val(:,1) = 1:200;
val(:,2) = real(d);
dlmwrite('evalues.txt',val,'delimiter','\t')
plotPattern(M, 5000, "k.eps")

M = readmatrix('L.txt');
M = sparse(M(:,1),M(:,2),M(:,3));
plotPattern(M, 5000, "L.eps")

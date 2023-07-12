%format long

A = readmatrix("M5.txt");

d = det(A);

Ainv = inv(A);

sprintf('%40.3f',d)

writematrix(Ainv, 'matlabInvM5.txt');
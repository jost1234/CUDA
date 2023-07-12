%matrix = randi([-10,9],75,75);

size = 32;

matrix = 20.*rand(size,size,'double') -10;

invMatrix = inv(matrix);

writematrix(matrix, 'M32.txt');
writematrix(invMatrix, 'matlabInvM32.txt');
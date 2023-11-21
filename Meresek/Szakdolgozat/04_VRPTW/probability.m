p = vpa(1-0.55^50);
ants = 20480;
rep = 70;
randomGenerations = 500000;
n = ants*rep*randomGenerations;

P = 1-(p)^n

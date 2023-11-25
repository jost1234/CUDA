data = readmatrix('CVRP_Grafikonok.xlsx');

size = data(:,1);

error10 = data(:,5);
time10 = data(:,6);

error30 = data(:,9);
time30 = data(:,10);

error50 = data(:,13);
time50 = data(:,14);

figure;
plot(size,error10,'b-x', size,error30,'r-x',size,error50,'g-x');
xlabel('Gráfcsúcsok száma','FontSize',20);
ylabel (' Átlagos hiba (%)','FontSize',20);
title('Százalékos eltérés az optimumtól','FontSize',20);
lgd = legend('10 rep', '30 rep', '50 rep');
fontsize(lgd,14,'points')
xlim([0,75])
ylim([0,300]);
%%
figure;
plot(size,time1,'b-x', size,time2,'r-x');


xlabel('Gráfcsúcsok száma','FontSize',20);
ylabel (' Futásidő (sec)','FontSize',20);
title('Futásidő mérések','FontSize',20);
lgd = legend('TSP v1', 'TSP v2');
fontsize(lgd,14,'points')
xlim([0,50])
ylim([0,160]);
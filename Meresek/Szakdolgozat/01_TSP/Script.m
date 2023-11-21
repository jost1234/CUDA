data = readmatrix('Grafikonok.xlsx');

size = data(:,1);

error1 = data(:,4);
time1 = data(:,5);

error2 = data(:,7);
time2 = data(:,8);
 
figure;
plot(size,error1,'b-x', size,error2,'r-x');


xlabel('Gráfcsúcsok száma','FontSize',20);
ylabel (' Átlagos hiba (%)','FontSize',20);
title('Százalékos eltérés az optimumtól','FontSize',20);
lgd = legend('TSP v1', 'TSP v2');
fontsize(lgd,14,'points')
xlim([0,50])
ylim([0,50]);

figure;
plot(size,time1,'b-x', size,time2,'r-x');


xlabel('Gráfcsúcsok száma','FontSize',20);
ylabel (' Futásidő (sec)','FontSize',20);
title('Futásidő mérések','FontSize',20);
lgd = legend('TSP v1', 'TSP v2');
fontsize(lgd,14,'points')
xlim([0,50])
ylim([0,160]);
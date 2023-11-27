% TSP

data = readmatrix('Grafikonok.xlsx');

TSPsize = data(1:5,1);

TSPerror1 = data(1:5,4);
TSPtime1 = data(1:5,5);

TSPerror2 = data(1:5,7);
TSPtime2 = data(1:5,8);
 
f = figure;
f.Position = [200 200 640 480];
plot(TSPsize,TSPerror1,'b-x', TSPsize,TSPerror2,'r-x');


xlabel('Gráfcsúcsok száma','FontSize',30);
ylabel (' Átlagos hiba (%)','FontSize',30);
ttl = title('Százalékos eltérés az optimumtól','FontSize',30);
lgd = legend({'TSP v1', 'TSP v2'},'Location','northwest');
fontsize(lgd,18,'points')
fontsize(gca, 18,'points')
fontsize(ttl, 25,'points')
xlim([0,50])
ylim([0,50]);

f = figure;
f.Position = [840 200 640 480];
plot(TSPsize,TSPtime1,'b-x', TSPsize,TSPtime2,'r-x');


xlabel('Gráfcsúcsok száma','FontSize',30);
ylabel (' Futásidő (sec)','FontSize',30);
ttl = title('Futásidő mérések','FontSize',30);
lgd = legend({'TSP v1', 'TSP v2'},'Location','northwest');
fontsize(lgd,18,'points')
fontsize(gca, 18,'points')
fontsize(ttl, 25,'points')
xlim([0,50])
ylim([0,160]);
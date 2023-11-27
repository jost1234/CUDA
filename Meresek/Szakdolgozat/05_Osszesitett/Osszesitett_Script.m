% Osszesitett

data = readmatrix('Grafikonok.xlsx');

%TSP
TSPsize = data(4:8,1);

TSPerror30 = data(4:8,7);
TSPtime30 = data(4:8,8);
 
%CVRP
Csize = data(9:13,1);

Cerror30 = data(9:13,9);
Ctime30 = data(9:13,10);

%CVRPTW

TWsize = data(1:3,1);
TWerror30 = data(1:3,11);
TWtime30 = data(1:3,12);

f = figure;
f.Position = [200 200 640 480];
plot(TSPsize,TSPerror30,'b-x', Csize,Cerror30,'r-x',TWsize,TWerror30,'g-x');


xlabel('Gráfcsúcsok száma','FontSize',30);
ylabel (' Átlagos hiba (%)','FontSize',30);
ttl = title('Százalékos eltérés az optimumtól','FontSize',30);
lgd = legend({'TSP v2', 'CVRP','CVRPTW'},'Location','northwest');
fontsize(lgd,18,'points')
fontsize(gca, 18,'points')
fontsize(ttl, 25,'points')
xlim([0,105])
ylim([0,400]);
%%
f = figure;
f.Position = [840 200 640 480];
plot(TSPsize,TSPtime30,'b-x', Csize,Ctime30,'r-x',TWsize,TWtime30,'g-x');


xlabel('Gráfcsúcsok száma','FontSize',30);
ylabel (' Futásidő (sec)','FontSize',30);
ttl = title('Futásidő mérések','FontSize',30);
lgd = legend({'TSP v2', 'CVRP','CVRPTW'},'Location','northwest');
fontsize(lgd,18,'points')
fontsize(gca, 18,'points')
fontsize(ttl, 25,'points')
xlim([0,105])
ylim([0,1510]);
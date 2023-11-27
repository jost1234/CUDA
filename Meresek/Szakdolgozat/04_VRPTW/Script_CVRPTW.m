% CVRPTW

data = readmatrix('VRPTW_Grafikonok.xlsx');

TWsize = data(:,1);

TWerror10 = data(:,6);
TWtime10 = data(:,7);

TWerror30 = data(:,11);
TWtime30 = data(:,12);

TWerror50 = data(:,16);
TWtime50 = data(:,17);

f = figure;
f.Position = [200 200 640 480];
plot(TWsize,TWerror10,'b-x', TWsize,TWerror30,'r-x',TWsize,TWerror50,'g-x');
xlabel('Gráfcsúcsok száma','FontSize',30);
ylabel (' Átlagos hiba (%)','FontSize',30);
ttl = title('CVRPTW Százalékos eltérés','FontSize',30);
lgd = legend({'10 rep', '30 rep', '50 rep'},'Location','northwest');
fontsize(lgd,18,'points')
fontsize(gca, 18,'points')
fontsize(ttl, 25,'points')
xlim([0,105])
ylim([0,600]);

f = figure;
f.Position = [840 200 640 480];
plot(TWsize,TWtime10,'b-x', TWsize,TWtime30,'r-x',TWsize,TWtime50,'g-x');


xlabel('Gráfcsúcsok száma','FontSize',30);
ylabel (' Futásidő (sec)','FontSize',30);
ttl = title('Futásidő mérések','FontSize',30);
lgd = legend({'10 rep', '30 rep', '50 rep'},'Location','northwest');
fontsize(lgd,18,'points')
fontsize(gca, 18,'points')
fontsize(ttl, 25,'points')
xlim([0,105])
ylim([0,2600]);
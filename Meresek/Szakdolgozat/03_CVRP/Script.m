% CVRP

data = readmatrix('CVRP_Grafikonok.xlsx');

Csize = data(:,1);

error10 = data(:,5);
time10 = data(:,6);

Cerror30 = data(:,9);
Ctime30 = data(:,10);

error50 = data(:,13);
time50 = data(:,14);

f = figure;
f.Position = [200 200 640 480];
plot(Csize,error10,'b-x', Csize,Cerror30,'r-x',Csize,error50,'g-x');
xlabel('Gráfcsúcsok száma','FontSize',30);
ylabel (' Átlagos hiba (%)','FontSize',30);
ttl = title('CVRP Százalékos eltérés','FontSize',30);
lgd = legend({'10 rep', '30 rep', '50 rep'},'Location','northwest');
fontsize(lgd,18,'points')
fontsize(gca, 18,'points')
fontsize(ttl, 25,'points')
xlim([0,75])
ylim([0,300]);
%%
f = figure;
f.Position = [840 200 640 480];
plot(Csize,time10,'b-x', Csize,Ctime30,'r-x',Csize,time50,'g-x');


xlabel('Gráfcsúcsok száma','FontSize',30);
ylabel (' Futásidő (sec)','FontSize',30);
ttl = title('Futásidő mérések','FontSize',30);
lgd = legend({'10 rep', '30 rep', '50 rep'},'Location','northwest');
fontsize(lgd,18,'points')
fontsize(gca, 18,'points')
fontsize(ttl, 25,'points')
xlim([0,75])
ylim([0,600]);
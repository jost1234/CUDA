data = readmatrix('Grafikonok.xlsx');

size = data(:,1);
error = data(:,4);

figure;
plot(size,error,'b-x');
xlabel('Gráfcsúcsok száma');
ylabel (' Átlagos hiba (%)');
title('Átlagos eltérés a minimumtól');
xlim([size(1),50])
ylim([error(1),52]);
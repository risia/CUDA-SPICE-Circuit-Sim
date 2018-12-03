data = csvread("transient.csv", 1, 0);
x = data(:,1);
v1 = data(:,2);
v2 = data(:,3);
v3 = data(:,4);

figure
plot(x, v1);
xlabel('Time (s)');
ylabel('V1');

hold on

plot(x, v2);
plot(x, v3);

hold off
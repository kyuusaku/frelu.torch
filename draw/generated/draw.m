% draw

%%
%(exp(a*x+b)*(a*x+b)+exp(c*x+d)*(c*x+d))/(exp(a*x+b)+exp(c*x+d))
cftool;

%%
x = -10:0.1:10;
x_plot = -10:0.5:10;

%%
y_relu = relu(x);
a = 0.02791; b = 0.2268; c = 0.9721; d = 0.2268;
y_relu = relu(x_plot);
[y_welu,l1,l2] = welu(x,a,b,c,d);
h = figure;
set(gca,'FontSize',20);
plot(x_plot,y_relu,':*k','MarkerSize',8,'LineWidth',2);
hold on;
plot(x,y_welu,'-','LineWidth',2,'Color',[0,0.5,0.85]);
grid on;
legend('ReLU','WeLU','Location','SouthEast');
print(h,'relu.png','-dpng');
close(h);

%%
k=0.2;
y_prelu = prelu(x,k);
a = 0.2274; b = 0.2468; c = 0.9726; d = 0.2469;
y_prelu = prelu(x_plot,k);
[y_welu,l1,l2] = welu(x,a,b,c,d);
h = figure;
set(gca,'FontSize',20);
plot(x_plot,y_prelu,':*k','MarkerSize',8,'LineWidth',2);
hold on;
plot(x,y_welu,'-','LineWidth',2,'Color',[0,0.5,0.85]);
grid on;
legend('PReLU(0.2)','WeLU','Location','SouthEast');
print(h,'prelu.png','-dpng');
close(h);

%%
y_elu = elu(x);
a = 0.05361; b = -0.5664; c = 0.9684; d = 0.2375;
y_elu = elu(x_plot);
[y_welu,l1,l2] = welu(x,a,b,c,d);
h = figure;
set(gca,'FontSize',20);
plot(x_plot,y_elu,':*k','MarkerSize',8,'LineWidth',2);
hold on;
plot(x,y_welu,'-','LineWidth',2,'Color',[0,0.5,0.85]);
grid on;
legend('ELU','WeLU','Location','SouthEast');
print(h,'elu.png','-dpng');
close(h);

%%
y_maxout = abs(x);
a = -0.9807; b = 0.1387; c = 0.9807; d = 0.1387;
y_maxout = abs(x_plot);
[y_welu,l1,l2] = welu(x,a,b,c,d);
h = figure;
set(gca,'FontSize',20);
plot(x_plot,y_maxout,':*k','MarkerSize',8,'LineWidth',2);
hold on;
plot(x,y_welu,'-','LineWidth',2,'Color',[0,0.5,0.85]);
grid on;
legend('MaxOut','WeLU','Location','SouthEast');
print(h,'maxout.png','-dpng');
close(h);

%%
y_pelu  = pelu( x , 1, 0.1);
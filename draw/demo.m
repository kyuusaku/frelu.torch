clc;
clear;
close all;
figure1 = figure;

% ´´½¨ axes
axes1 = axes('Parent',figure1,'YGrid','on','XGrid','on');
box(axes1,'on');
hold(axes1,'all');


hold on;

plotlog('resnet-56.log','g');
plotlog('preresnet-elu_b_3x3_end-0.0002.log','b');
%plotlog('resnet-welu-56-0.0005.log','r');
plotlog('resnet-pelu-0.001-110.log','c');
%plotlog('resnet-welu-56-0.001.log','m');
%plotlog('resnet-welu-56-0.0002-b-lowlr0.1.log','m');
plotlog('resnet-welu-56-0.0004-0-b--1.log','m');
%plotlog('resnet-welu-56-0.0002-0-b-0-wlast.log','m');
%plotlog('resnet-welu-56-0.0009-0-b--1.log','r');
%plotlog('resnet-welu-56-0.001-0-b--1.log','r');
plotlog('resnet-welu-56-0.0002-0.0002-b-0.log','r');
%plotlog('resnet-welu-56-0.0002-0.0002-b-0-wlast.log','r');
%plotlog('resnet-welu-56-0.0004-0.0002.log','r');
%plotlog('resnet-welu-56-0.001-0.0002.log','r');
%plotlog({'train-2016-12-08-10-37-46.log','train-2016-12-08-11-37-34.log','train-2016-12-08-14-02-21.log','train-2016-12-08-14-44-15.log'},'m');
plotlog('resnet-welu-b-0.0002-56.log','k');



legend(axes1,'show');
%xlim(axes1,[100 200]);

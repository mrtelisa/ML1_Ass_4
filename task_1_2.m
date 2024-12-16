clear all
close all
clc

%% TASK 1

load glass_dataset

x = glassInputs;
t = glassTargets;

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% View the Network
view(net)

% Plots
figure, plotperform(tr)
figure, plottrainstate(tr)
figure, ploterrhist(e)
figure, plotconfusion(t,y)
figure, plotroc(t,y)

%% TASK 2

% Select a random class from 1 to 10
cl1 = randi([1,10]);
cl2 = randi([1,10]);

while cl1 == cl2
    cl2 = randi([1,10]);
end

% Load data
[train_set, train_cl] = loadMNIST(0,[cl1,cl2]);

% Select number of hidden units
nh = 2;

% Train the autoencoder
myAutoencoder = trainAutoencoder(train_set',nh);
myEncodedData = encode(myAutoencoder,train_set');

%% Plot encoded data
plotcl(myEncodedData',train_cl)



clear all
close all
clc

%% Select a random class from 1 to 10
cl1 = randi([1,10]);
cl2 = randi([1,10]);

while cl1 == cl2
    cl2 = randi([1,10]);
end

% Load data
[train_set, train_cl] = loadMNIST(0,[cl1,cl2]);

% Select number of hidden units
nh = 2;

%% Train the autoencoder
myAutoencoder = trainAutoencoder(train_set',nh);
myEncodedData = encode(myAutoencoder,train_set');

%% Plot encoded data
plotcl(myEncodedData',train_cl)



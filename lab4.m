close all
clear all
clc

%% Select a random class
% Choose between class 1 and 10
cl1 = randi([1, 10]);
cl2 = randi([1, 10]);

% I want two different classes
while cl1 == cl2
    cl2 = randi([1, 20]);
end

% Loading the data
[train_set, train_cl] = loadMNIST(0, [cl1, cl2]);

% Indicate the number of hidden units
nh = 2;

% Train the autoencoder
myAutoencoder = trainAutoencoder(train_set,nh);
myEncodedData = encode(myAutoencoder,train_set);

% Plot the encoded_data
plot_cl(myEncodedData', train_cl)
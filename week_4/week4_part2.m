clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
% (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
load('ex3data1.mat');

m = size(X, 1);

sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

%% ================ Part 2: Loading Pameters ================

fprintf('\nLoading Saved Neural Network Parameters ...\n')
load('ex3weights.mat');
% The matrices Theta1 and Theta2 will now loaded automatically
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26


%% ================= Part 3: Implement Predict =================
function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);

% My Code

% Getting the input layer -> a1
% Then making it vector
a1 = [ones(m, 1) X];

% Multiplying it with the weight
z2 = a1 * theta'

% Sigmoid of z2 will give us the hidden layer
a2 = sigmoid(z2)

% Doing same thing with this layer
m2 = size(a2, 1)
a2 = [ones(m2, 1) a2]
z3 = a3 * theta2'
a3 = sigmoid(z3)

% Now calculating the max from the last layer whose index corresponds to the correct digit
[val, index] = max(a3, [], 2)

p = index
end

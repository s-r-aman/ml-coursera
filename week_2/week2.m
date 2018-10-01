clear ; close all; clc

data = load('data1.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
theta = zeros(3, 1);

mean = sum(X(: , 1)) / m;

sd = std(X(: , 1));
% sd = max(X(: , 1)) - min(X(: , 1));

dataset_X = [ones(m, 1), (X(:, 1).- mean) / sd, X(:, 2)];

function cost = costMulti(X, theta, y, m)
  cost = (1 / (2 * m)) * sum(( X * theta - y).^2); 
endfunction

function gradientDescentMulti
  %theta = theta - alpha * (1 / m) * sum(( X * theta - y).X()); 
endfunction
j = costMulti(dataset_X, theta, y, m);

p =  dataset_X * theta - y

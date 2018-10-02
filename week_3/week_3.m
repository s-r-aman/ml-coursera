clear ; close all; clc

% Defining and Loading Data Set

data = load('data.txt');

X = data(:, 1:2);
y = data(:, 3);

[m, n] = size(X);;


% Making Different Functions
function plotData(X, y)
figure; hold on;

pos = find(y==1);
neg = find(y == 0);

plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7)
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)

hold off;

end
initial_theta = zeros(n + 1, 1);

function g = sigmoid(z)
   g = 1 / (1 + exp(-z));
endfunction

function j = costFunction(X, y, m, theta)
  j = (1/m) * sum( -y' * log((sigmoid(X * theta)'))  - (1 - y)' * log(1 - (sigmoid(X * theta)')));
endfunction


fprintf('Plotting the data. \n');
fprintf('Press Enter to continue. \n');
plotData(X, y);
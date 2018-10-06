clear ; close all; clc

data = load('data1.txt')
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

function J = costFunction(theta, X, y)
  m = length(y)
  J = (1/(2*m)) * sum((theta' * X' - y).^2)
end



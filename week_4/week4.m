clear ; close all; clc

% Loading the data
load('data1.mat')

m = size(X, 1);

rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);



function [J, grad] = lrCostFunction(theta, X, y, lambda)
m = length(y);
J = 0;
grad = zeros(size(theta));

term1 = (1 / m) * sum( -y' * log(sigmoid(X * theta))- (1 - y)' * log(1-sigmoid(X * theta)));
term2 = (lambda/(2 * m)) * sum(theta.^2);

J = ( (1 / m) * sum(-y'*log(sigmoid(X*theta)) - (1-y)'*log( 1 - sigmoid(X*theta))) ) ;

J = term1 + term2;

grad = grad(:);
 
end

fprintf('\nTesting lrCostFunction() with regularization');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t)
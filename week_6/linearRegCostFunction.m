function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
m = length(y);
J = 0;
grad = zeros(size(theta));
J = (1/(2*m))*sum(power((X*theta - y),2))+ (lambda/(2*m)) * sum(power(theta(2:end),2));

G = (lambda/m) .* theta;
G(1) = 0; % this is always 0

grad = ((1/m) .* X' * (X*theta - y)) + G;
grad = grad(:);

end

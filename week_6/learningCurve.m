function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);
for i = 1:m
    X_sub = X(1:i, :);
    y_sub = y(1:i); 

    theta = trainLinearReg(X_sub, y_sub, lambda);

    error_train(i) = linearRegCostFunction(X_sub, y_sub, theta, 0);
    error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end
end

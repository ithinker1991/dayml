function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples


% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

z = X * theta;
h = sigmoid(z);

d = ones(size(theta),1);
d(1) = 0;

J1 = -y' * log(h);
J2 = -(1 - y') * log(1 - h);
t1 = [0;theta(2:end)];

J = (J1 + J2) / m + (lambda / (2*m)) * t1' * t1;
grad = zeros(size(theta));

grad = (X' * (h - y)) / m + lambda / m .* t1 ;  % n * 1

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end

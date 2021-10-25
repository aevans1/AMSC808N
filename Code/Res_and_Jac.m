%%
function [r,J] = Res_and_Jac(X,y,I,w)
% vector of residuals
exp_quad = exp(-myquadratic(X,y,I,w));
r = log(1 + exp_quad);

% the Jacobian matrix

% partial derivative of r with respect to q(x;w)
a = -exp_quad./(1+exp_quad);
[n,d] = size(X);
d2 = d^2;
ya = y.*a;

% partial derivative of q(x;w) with respect to w:
% 3 cases: 1) partial deriv for matrix W (quadratic term)
%          2) patial deriv for vector v (linear term)
%          3) partial deriv for vector b (constant term)

% Case 1) quadratic term partial derivs
qterm = zeros(n,d2);
for k = 1 : n
    xk = X(k,:); % row vector x
    xx = xk'*xk;
    qterm(k,:) = xx(:)';
end
% Case 2) linear term derivs: collected all together this returns X
% Case 3) constant term derivs: collected all together this returns 1
% identically

% collect all partial derivs from 1), 2), 3)
Y = [qterm,X,ones(n,1)];

% combine with partial deriv of r with respec to q, and with partial f
J = (ya*ones(1,d2+d+1)).*Y;
end
%%
function q = myquadratic(X,label,I,w)
X = X(I,:); % only evaluate quadratic at given data indices I

[N, d] = size(X);
d2 = d^2;
y = label(I);
W = reshape(w(1:d2),[d,d]);
v = w(d2+1:d2+d);
b = w(end);
qterm = zeros(N,1);
for i=1:N
     qterm(i) = y(i)*X(i,:)*W*X(i,:)';
end
 q = qterm + ((y*ones(1,d)).*X)*v + y*b;
end




function [w, f_list, gnorm_list] = LevenburgMarquardt(fun,X,y,w,kmax, tol)

%% Trust Region Parameters
Rmin = 0.1;     % minimum trust region radius
Rmax = 10.0;     % max trust region radius
R = 0.2*Rmax;   % initial trust region radius
threshold = 0.1;   % threshold for accepting proposed step

%% Optimizer Parameters
[N, dim] = size(X);
I = 1:N;
%% Begin optimizing

% Create list of gradient and function evals for tracking convergence
gnorm_list = zeros(1, 1000);
kmax
f_list = zeros(kmax + 1, 1);
f = fun(I, w);

[r, J] = Res_and_Jac(X, y, I, w);    
g = J'*r;

gnorm = norm(g);
gnorm_list(1) = gnorm;

iter = 1;
while (gnorm > tol) && iter < kmax
    
    % Find direction
    [r, J] = Res_and_Jac(X, y, I, w);
    p = finddirection(r, J, R);

    % Make step
    wnew = w + p;
    fnew = fun(I, wnew);
    m = 0.5*norm(J*p + r)^2;
    % Compute trust region ratio, compute new trust region
    ratio = (f - fnew)/(f - m);
    if ratio < 0.25
       fprintf("decreasing R");
       R = 0.25*R;
    elseif ((ratio > 0.75) && (norm(p) - R < 1e-6))
        fprintf("increasing R");
        R = min(2*R, Rmax);
    end

    % Check if accepting step or not
    ratio
    if ratio > threshold
       w = wnew;
       f = fun(I, w);
       [r, J] = Res_and_Jac(X, y, I, w);    
       g = J'*r;
%        fprintf("new step!");
    end
       
    gnorm = norm(g);
    gnorm_list(iter + 1) = gnorm;
    f_list(iter + 1) = f;
    iter = iter + 1
    gnorm
    
end
end

function p = finddirection(r, J, R)
B = J'*J;
[n, ~] = size(B);
B = B + eye(n)*10^(-6); %regularize
g = J'*r;
p_sol = -B\g;
if norm(p_sol) <= R
    % Accept unconstrained minimizer
    p = p_sol;
else
    % Solve constrained minimization problem
    lambda = 1;
    while true
        Blambda = B + lambda*eye(n);
        C = chol(Blambda);
        p = -C\(C'\g);
        pnorm = norm(p);
        if abs(pnorm - R) < 1e-6
%             fprintf("solved constrained part");
            break;
        end
        q = C'\p;
%         q = sqrt(Blambda)\p;
        qnorm = norm(q);
        phi = (1/R) - 1/norm(p);
        dphi =  -(norm(q)^2)/(norm(p)^3);
        lambda_new = lambda - (phi/dphi);
%         lambda_new = lambda + ((pnorm/qnorm)^2)*(pnorm - R)/R;
        if lambda_new < 0
            lambda = 0.5*lambda;
        else
            lambda = lambda_new;
        end
    end
end
end
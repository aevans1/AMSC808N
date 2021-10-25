function [w, f, gnorm_list] = GaussNewton(fun,X,y,w,kmax, tol)

%% Linesearch Parameters
gamma = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gamma)); % max # of iterations in line search
eta = 0.5; % backtracking stopping criterion factor

%% Optimizer Parameters
[N, dim] = size(X);
I = 1:N;
%% Begin optimizing
gnorm_list = zeros(1, 1000);
f = zeros(kmax + 1, 1); % f(i) is loss at step (i)
[r,J] = Res_and_Jac(X,y,I,w);
g = J'*r;
gnorm = norm(g);
gnorm_list(1) = gnorm;

% Find stepsize
alpha = linesearch(I,w,-g,g,fun,eta,gamma,jmax);

% Find direction
p = -g;

% Make step
wnew = w + alpha*p;
[r,J] = Res_and_Jac(X,y,I,wnew);
gnew = J'*r; 

% Update
w = wnew;
g = gnew;
gnorm = norm(g);
gnorm_list(2) = gnorm;
%% Start main loop
iter = 1;
while gnorm > tol

    % Find direction
    [r,J] = Res_and_Jac(X,y,I,w);
    p = finddirection(r, J);
    % Find stepsize
    [alpha,j] = linesearch(I,w,p,g,fun,eta,gamma,jmax);
    if j == jmax
        fprintf("taking a maximum step for linesearch");
        p = -g;
        [alpha,j] = linesearch(I,w,p,g,fun,eta,gamma,jmax);
    end
    % Make step
    wnew = w + alpha*p;
    [r,J] = Res_and_Jac(X,y,I,wnew);
    gnew = J'*r;
    
    % Update
    w = wnew;
    g = gnew;
    gnorm = norm(g);
    gnorm_list(iter+1) = gnorm;
    iter = iter + 1
    gnorm
    f(iter+1) = fun(I, w);
end

end
%% 
    function p = finddirection(r, J)
        M = J'*J;
        [n, ~] = size(M);
        M = M + eye(n)*10^(-6); %regularize
        g = J'*r;
        p = -M\g;
%         p = lsqminnorm(J, -r);
    end
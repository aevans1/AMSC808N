function [w, f, normgrad] = my_LBFGS(fun, gfun, X, y, w, bsz, kmax, tol)
% Separate data into: inputs, targets
[N, dim] = size(X);
f = zeros(kmax + 1, 1); % f(i) is loss at step (i)
normgrad = zeros(kmax, 1);
m = 5; % the number of steps to keep in memory
M = 20; % number of update steps between hessian updates
bszH = 10*bsz;
%%  Linesearch stuff
gamma = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gamma)); % max # of iterations in line search
eta = 0.5; % backtracking stopping criterion factor
%% LBFGS parameter setup 
[params, ~] = size(w);
s = zeros(params,m);
y = zeros(params,m);
rho = zeros(1,m);
%% Step 0
IH = randperm(N, bszH);      % random index selection
Ig = IH(:,1:bsz);

g = gfun(Ig, w);
normgrad(1) = norm(g);      % update norm gradient list
f(1) = fun(Ig, w);

a = linesearch(Ig,w,-g,g,fun,eta,gamma,jmax);
wnew = w - a*g;
gnew = gfun(Ig, wnew);
s(:,1) = wnew - w;
y(:,1) = gnew - g;
rho(1) = 1/(s(:,1)'*y(:,1));
w = wnew;
g = gnew;
nor = norm(g);
normgrad(2) = nor;
f(2) = fun(Ig, w);
%%
iter = 1;
while (nor > tol) && iter < kmax    
    g = gfun(Ig,w);
    if iter < m
        % continue new direction for building vector-pair list
        I = 1 : iter;
        p = finddirection(g,s(:,I),y(:,I),rho(I));
    else
        % compute new direction for replacing oldest vector-pair
        p = finddirection(g,s,y,rho);
    end
    
    [a,j] = linesearch(Ig,w,p,g,fun,eta,gamma,jmax);
    if j == jmax
        p = -g;
        [a,j] = linesearch(Ig,w,p,g,fun,eta,gamma,jmax);
    end
    step = a*p;
    wnew = w + step;
    
    IH = randperm(N, bszH);
    Ig = IH(:, 1:bsz);
    
    if (mod(iter,M) == 0)
        gnewH = gfun(IH, wnew);
        gH = gfun(IH, w);
        % replace oldest (s,y) vector pair and associated rho step
        s = circshift(s,[0,1]); 
        y = circshift(y,[0,1]);
        rho = circshift(rho,[0,1]);
        s(:,1) = step;
        y(:,1) = gnewH - gH;
        rho(1) = 1/(step'*y(:,1));
    end
    
    gnew = gfun(Ig,wnew);
    % update parameter vector w, gradient vector b, gradient norm list
    w = wnew;
    g = gnew;
    nor = norm(g);
    iter = iter + 1;
    normgrad(iter+1) = nor;   
    f(iter +1) = fun(Ig, w);
end
end

%%
function [a,j] = linesearch(I,w,p,b,fun,eta,gamma,jmax)
    a = 1;
    f0 = fun(I,w);
    aux = eta*b'*p;
    for j = 0 : jmax
        wtry = w + a*p;
        f1 = fun(I, wtry);
        if f1 < f0 + a*aux
            break;
        else
            a = a*gamma;
        end
    end
end
%%
    
function p = finddirection(b,s,y,rho)
% input: b = gradient dim-by-1
% s = matrix dim-by-m, s(:,i) = x_{k-i+1}-x_{k-i}
% y = matrix dim-by-m, y(:,i) = g_{k-i+1}-g_{k-i}
% rho is 1-by-m, rho(i) = 1/(s(:,i)'*y(:,i))
m = size(s,2);
a = zeros(m,1);  
for i = 1 : m
    a(i) = rho(i)*s(:,i)'*b;
    b = b - a(i)*y(:,i);
end
gamma = s(:,1)'*y(:,1)/(y(:,1)'*y(:,1)); % H0 = gamma*eye(dim)
b = b*gamma;
for i = m :-1 : 1
    aux = rho(i)*y(:,i)'*b;
    b = b + (a(i) - aux)*s(:,i);
end
p = -b;
end
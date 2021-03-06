function [w, f, normgrad, a] = SLBFGS(fun, gfun, X, y, w, max_epochs, bszg, bszH, M)
[N, dim] = size(X);
num_batches = ceil(N/bszg);
update_freq = ceil(num_batches/5);     % update 5 times per epoch
iter = 1;
f = [];
normgrad = [];


%%  Linesearch stuff
gamma = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gamma)); % max # of iterations in line search
eta = 0.5; % backtracking stopping criterion factor
%% LBFGS parameter setup 
m = 5; % the number of steps to keep in memory

[params, ~] = size(w);
s = zeros(params,m);
y = zeros(params,m);
rho = zeros(1,m);

%% Step 0: Getting a good initial step before the full iteration
IH = randperm(N, bszH);      % random index selection
Ig = IH(:,1:bszg);

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
iter = 3;

 %%
for i = 1 : max_epochs
    for k = 1 : num_batches
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

        IH = randperm(N, bszH);     % update at every batch, use every M batches
        Ig = IH(:, 1:bszg);          % update at every batch, via the Hessian batch

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
        if (mod(k, update_freq) == 0)
            f(end + 1) = fun(Ig, w);
            normgrad(end + 1) = norm(g);      % norm of gradient
        end
        iter = iter + 1;
    end
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
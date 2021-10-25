function [w, f, normgrad] = ADAM(fun, gfun, gfun2, Hvec, X, y, w, bsz, kmax, tol)
    % Separate data into: inputs, targets
    [N, dim] = size(X);
    f = zeros(kmax + 1, 1); % f(i) is loss at step (i)
    normgrad = zeros(kmax, 1);
    
    %%% For ADAM
    beta1 = 0.9;
    beta2 = 0.999;
    eps = 10^(-8);
    eta = 0.001;
    Ig = randperm(N, bsz);      % random index selection
    b = gfun(Ig, w);            % gradient of batch
    m = 0*b;
    v = 0*b;
    %%%
    for k = 1 : kmax
        Ig = randperm(N, bsz);      % random index selection
        % Apply model to inputs, get  model outputs
        % Apply loss function to model outputs and inputs
        % Find the new step direction for optimizer
        b = gfun(Ig, w);            % gradient of batch
        bsq = gfun2(Ig, w);
        
        normgrad(k) = norm(b);      % norm of gradient
        f(k) = fun(Ig, w);  
        % Update optimizer
        m = beta1*m + (1 - beta1)*b;        % first moments
        m = (1/(1 - beta1^k))*m;
        v = beta2*v + (1 - beta2)*(bsq);   % second moments
        v = (1/(1 - beta2^k))*v;
        maux = m./(sqrt(v) + eps);
        w = w - eta*maux;        
        
        if normgrad(k) < tol
            break;
        end
    end    
end
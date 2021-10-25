function [w, f, normgrad] = NGD(fun, gfun, Hvec, X, y, w, bsz, kmax, tol)
    % Separate data into: inputs, targets
    [N, dim] = size(X);
    f = zeros(kmax + 1, 1); % f(i) is loss at step (i)
    normgrad = zeros(kmax, 1);
    
    %%% For Nesterov
    x = w;                  % intial `position' in Nesterov steps
    lambda = 0;
    %%%
    for k = 1 : kmax
        Ig = randperm(N, bsz);      % random index selection
        
        % Apply model to inputs, get  model outputs
        % Apply loss function to model outputs and inputs
        % Find the new step direction for optimizer
        b = gfun(Ig, x);            % gradient of batch
        normgrad(k) = norm(b);      % norm of gradient
        f(k) = fun(Ig, x);
        
        % Update optimizer
        stepsize = 0.001;           % step for gradient
        wnew = x - stepsize*b;
        lambda_new = 0.5*(1 + sqrt(1 + 4*lambda^2));
        gamma = (1 - lambda)/lambda_new;
        lambda = lambda_new;
        x = (1 - gamma)*wnew + gamma*w;
        w = wnew;
        if normgrad(k) < tol
            break;
        end
    end    
end
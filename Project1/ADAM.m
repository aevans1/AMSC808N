function [w, f, normgrad, iter] = ADAM(fun, gfun, gfun2, Hvec, X, y, w, bsz, max_epochs, tol)

    [N, dim] = size(X);
    num_batches = ceil(N/bsz);
    update_freq = ceil(num_batches/5);     % update 5 times per epoch
    iter = 1;
    f = [];
    normgrad = [];
    

    %%% For ADAM
    beta1 = 0.9;
    beta2 = 0.9;
    eps = 10^(-8);
    eta = 0.001;
    Ig = randperm(N, bsz);      % random index selection
    b = gfun(Ig, w);            % gradient of batch
    m = 0*b;
    v = 0*b;
    %%%
    for i = 1 : max_epochs
        for j = 1 : num_batches
            Ig = randperm(N, bsz);      % random index selection
            % Apply model to inputs, get  model outputs
            % Apply loss function to model outputs and inputs
            % Find the new step direction for optimizer
            b = gfun(Ig, w);            % gradient of batch
%             bsq = gfun2(Ig, w);
            bsq = b.^2;

            if (mod(j, update_freq) == 0)
                f(end + 1) = fun(Ig, w);
                normgrad(end + 1) = norm(b);      % norm of gradient
            end

            % Update optimizer
            m = beta1*m + (1 - beta1)*b;        % first moments
            m = (1/(1 - beta1^iter))*m;
            v = beta2*v + (1 - beta2)*(bsq);   % second moments
            v = (1/(1 - beta2^iter))*v;
            maux = m./(sqrt(v) + eps);
            w = w - eta*maux;        

            iter = iter + 1;
        end
    end
end    

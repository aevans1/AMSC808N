function [w, f, normgrad, iter] = SGD(fun, gfun, Hvec, X, y, w, bsz, max_epochs, tol)
    
    % Separate data into: inputs, targets
    [N, dim] = size(X);
    num_batches = ceil(N/bsz);
    update_freq = ceil(num_batches/5);     % update 5 times per epoch
    iter = 1;
    f = [];
    normgrad = [];
    for i = 1 : max_epochs
        for j = 1 : num_batches
            Ig = randperm(N, bsz);      % random index selection
            
            % Apply model to inputs, get  model outputs
            % Apply loss function to model outputs and inputs
            % Find the new step direction for optimizer
            b = gfun(Ig, w);               % gradient of batch
            
            if (mod(j, update_freq) == 0)
                f(end + 1) = fun(Ig, w);
                normgrad(end + 1) = norm(b);      % norm of gradient
            end
            
            % Update optimizer
            stepsize = 0.1;
            w = w - stepsize*b;
            iter = iter + 1;    % track total number of parameter updates
        end
%         str = sprintf("finished with epoch: %d \n \n", i);
%         fprintf(str);
    end
end
function [w, f, normgrad, stepsize] = SGD(fun, gfun, Hvec, X, y, w, bsz, max_epochs, tol, step_strategy)
    
   
    
    %% Stepsize Strategy
    beta = 0.5;
    alpha = 0;
    
    if step_strategy == 1
        stepsize_func = @(i) beta/(alpha + i);
    elseif step_strategy == 2
        stepsize_func = @(i) beta/(alpha + 2^(i-1));
    end
    
    %%
    for i = 1 : max_epochs
        stepsize = stepsize_func(i);
        for j = 1 : num_batches
%             stepsize = stepsize_func(iter);
            Ig = randperm(N, bsz);      % random index selection
            
            % Apply model to inputs, get  model outputs
            % Apply loss function to model outputs and inputs
            % Find the new step direction for optimizer
            b = gfun(Ig, w);               % gradient of batch
            
            if (mod(j, update_freq) == 0)
                f(end + 1) = fun(Ig, w);
                normgrad(end + 1) = norm(b);      % norm of gradient
            end
            w = w - stepsize*b;
            iter = iter + 1;    % track total number of parameter updates
        end
%         str = sprintf("finished with epoch: %d \n \n", i);
%         fprintf(str);
    end
end
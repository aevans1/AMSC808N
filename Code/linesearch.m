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
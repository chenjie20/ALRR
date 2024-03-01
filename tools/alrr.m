function [Z, iter] = alrr(X, lambda, debug)

%The convergence of ALRR is theoretically guaranteed under certain conditions, 
%where ALRR requires just three iteration computations for optimization. 

% default parameters
tol = 1e-4;
max_iter = 3;
[m, n] = size(X);
P = eye(m);
Z = zeros(n, n);
iter = 0;

debug = 1;
if nargin < 3
    debug = 0;
end
if lambda <= 1
    disp('parameter error');
end

obj_value = 0;
while iter < max_iter;
    
    iter = iter + 1;
    obj_value_tmp = obj_value;
    
    D = P' * X;
    [~, s, V] = svd(D, 'econ');
    s = diag(s);
    lmd = 1 / sqrt(lambda);
    k = length(find(s > lmd));
    Z = V(:, 1 : k) * (eye(k) -  diag(1 / (lambda * (s(1 : k) .^ 2)))) * V(:, 1 : k)';
    
    E = X - X * Z;
    EE = E * E';
    [eigvector, eigvalue] = eig(EE, X * X');    
%     P1 = eigvector(:, 1 : k);
    [~, ind] = sort(diag(eigvalue), 'ascend');
    P = eigvector(:,ind(1 : k));
   
    if k <= 1
        disp('error');
        break;
    end
    
   if debug == 1
       [~, ss, ~]= svd(Z);
       part1 = sum(diag(ss));
       part2 = norm((P' * X - P' * X * Z), 'fro');
       obj_value = part1 + lambda / 2 * part2;
       rr = abs((obj_value - obj_value_tmp) / obj_value);
       if(rr < tol)
           disp([iter, rr]);
           break;
       end       
   end
end

end



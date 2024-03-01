close all;
clear;
clc;

addpath('data');
addpath('tools');

data = load('ExtendedYaleB.mat');
X = data.EYALEB_DATA;
gnd = data.EYALEB_LABEL';
K = max(gnd);

dim = 500;
if dim > 0
    [eigen_vector, eigen_value, mean_value] = f_pca(X, dim);
    X = eigen_vector' * X;
end

class_labels = zeros(1, K);
for idx =  1 : K
    class_labels(idx) = length(find(gnd == idx));
end

lambdas = [15];
betas = [2];

for lmd_idx = 1 : length(lambdas)
    lambda = lambdas(lmd_idx);   
       tic;
       [Z, iter] = alrr(normc(X), lambda);       
       for beta_idx = 1 : length(betas)
        beta = betas(beta_idx);

        [U, s, ~] = svd(Z);
        s = diag(s);
        r = sum(s>1e-6);

        U = U(:, 1 : r);
        s = diag(s(1 : r));

        M = U * s.^(1/2);
        mm = normr(M);
        rs = mm * mm';
        L = rs.^(2 * beta);
        time_cost = toc;

        actual_ids = spectral_clustering(L, K);
        acc = accuracy(gnd, actual_ids);

        cluster_data = cell(1, K);
        for pos_idx =  1 : K
            cluster_data(1, pos_idx) = { gnd(actual_ids == pos_idx)' };
        end 
        [nmi, purity, fmeasure, ri, ari] = calculate_results(class_labels, cluster_data);        
        disp([lambda, beta, acc, nmi, ari, time_cost, iter]);

        dlmwrite('alrr_face_yaleb.txt', [lambda, beta, acc, nmi, ari, time_cost, iter], '-append', 'delimiter', '\t', 'newline', 'pc');
     end
end



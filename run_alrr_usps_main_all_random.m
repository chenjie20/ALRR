close all;
clear;
clc;

addpath('data');
addpath('tools');

load('usps.mat');

K = 10;
X = mat2gray(data(:, 2 : end))';
n = size(X, 2);
gnd = data(1 : n, 1);

dim = 30;
if dim > 0
    [eigen_vector, eigen_value, mean_value] = f_pca(X, dim);
    X = eigen_vector' * X;
end

num_clusters_set = [3, 5, 8];
run_times = 10;
[m, n] = size(X);

lambdas = [2];
betas = [5];

clusters_set_len = length(num_clusters_set);

lambda_len = length(lambdas);
beta_len = length(betas);
accs = zeros(lambda_len, beta_len, clusters_set_len, run_times);
nmis = zeros(lambda_len, beta_len, clusters_set_len, run_times);
purities = zeros(lambda_len, beta_len, clusters_set_len, run_times);
fmeasures = zeros(lambda_len, beta_len, clusters_set_len, run_times);
ris = zeros(lambda_len, beta_len, clusters_set_len, run_times);
aris = zeros(lambda_len, beta_len, clusters_set_len, run_times);
time_costs = zeros(lambda_len, beta_len, clusters_set_len, run_times);

for lmd_idx = 1 : length(lambdas)
    lambda = lambdas(lmd_idx); 
    for cluster_set_idx = 1 : length(num_clusters_set)
        num_clusters = num_clusters_set(cluster_set_idx);
        for run_index = 1 : run_times
            rand('state', (cluster_set_idx - 1) * num_clusters + run_index);
            clusters_set = randperm(K, num_clusters);
            num_samples = 0;
            for cluster_idex = 1 : length(clusters_set)
                cluster_label = clusters_set(cluster_idex);
                num_samples = num_samples + length(find(gnd == cluster_label));
            end
            XX = zeros(m, num_samples);
            new_gnd = zeros(num_samples, 1);
            start_index = 1;
            for cluster_idex = 1 : length(clusters_set)        
                cluster_label = clusters_set(cluster_idex);
                len = length(find(gnd == cluster_label));
                new_gnd(start_index : (start_index + len - 1), 1) = gnd(gnd == cluster_label, 1);
                XX(:, start_index : (start_index + len - 1)) = X(:, (gnd == cluster_label));
                start_index = start_index + len; 
            end
            [new_labels, ~] = refresh_labels(new_gnd, K);

            class_labels = zeros(1, num_clusters);
            for idx =  1 : num_clusters
                class_labels(idx) = length(find(new_labels == idx));
            end
            tic;
            [Z, iter] = alrr(normc(XX), lambda);       
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
                time_costs(lmd_idx, beta_idx, cluster_set_idx, run_index) = time_cost;

                actual_ids = spectral_clustering(L, num_clusters);
                acc = accuracy(new_labels, actual_ids);
                accs(lmd_idx, beta_idx, cluster_set_idx, run_index) = acc;

                cluster_data = cell(1, num_clusters);
                for pos_idx =  1 : num_clusters
                    cluster_data(1, pos_idx) = { new_labels(actual_ids == pos_idx)' };
                end 
                [nmi, purity, fmeasure, ri, ari] = calculate_results(class_labels, cluster_data); 
                nmis(lmd_idx, beta_idx, cluster_set_idx, run_index) = nmi;
                purities(lmd_idx, beta_idx, cluster_set_idx, run_index) = purity;
                fmeasures(lmd_idx, beta_idx, cluster_set_idx, run_index) = fmeasure;
                ris(lmd_idx, beta_idx, cluster_set_idx, run_index) = ri;
                aris(lmd_idx, beta_idx, cluster_set_idx, run_index) = ari;
                disp([lambda, beta, cluster_set_idx, run_index, acc, nmi, ari, purity, fmeasure, time_cost, iter]);
                dlmwrite('alrr_usps_random.txt', [lambda, beta, cluster_set_idx, run_index, acc, nmi, ari, purity, fmeasure, time_cost, iter], '-append', 'delimiter', '\t', 'newline', 'pc');
            end
        end
        for beta_idx = 1 : length(betas)
            beta = betas(beta_idx);
            avg_acc = mean(accs(lmd_idx, beta_idx , cluster_set_idx, :));
            std_acc = std(accs(lmd_idx, beta_idx , cluster_set_idx, :));
            avg_nmi = mean(nmis(lmd_idx, beta_idx , cluster_set_idx, :));
            std_nmi  = std(nmis(lmd_idx, beta_idx , cluster_set_idx, :));
            avg_puritiy = mean(purities(lmd_idx, beta_idx , cluster_set_idx, :));
            std_puritiy  = std(purities(lmd_idx, beta_idx , cluster_set_idx, :));
            avg_fmeasure = mean(fmeasures(lmd_idx, beta_idx , cluster_set_idx, :));
            std_fmeasure  = std(fmeasures(lmd_idx, beta_idx , cluster_set_idx, :));            
            avg_ari = mean(aris(lmd_idx, beta_idx , cluster_set_idx, :));
            std_ari  = std(aris(lmd_idx, beta_idx , cluster_set_idx, :));
            avg_time_cost = mean(time_costs(lmd_idx, beta_idx , cluster_set_idx, :));
            std_time_cost  = std(time_costs(lmd_idx, beta_idx , cluster_set_idx, :));
            dlmwrite('alrr_usps_random_avg.txt', [lambda, beta, cluster_set_idx, avg_acc, avg_nmi, avg_puritiy, avg_fmeasure, avg_ari, std_acc, std_nmi, std_puritiy, std_fmeasure, std_ari, avg_time_cost, std_time_cost], '-append', 'delimiter', '\t', 'newline', 'pc');
        end
    end
end

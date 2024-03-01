close all;
clear;
clc;

addpath('data');
addpath('tools');

load('COIL100.mat');
XX = im2double(fea');
K = max(gnd);
gnd = gnd';
[row_size, col_size] = size(XX);

X = zeros(row_size, col_size);
noise_ratios = [0.1, 0.2];
noise_len = length(noise_ratios);
run_times = 10;

class_labels = zeros(1, K);
for idx =  1 : K
    class_labels(idx) = length(find(gnd == idx));
end
        
lambdas =[1.5];
betas = [6];
lambda_len = length(lambdas);
beta_len = length(betas);

accs = zeros(noise_len, run_times, lambda_len, beta_len);
nmis = zeros(noise_len, run_times, lambda_len, beta_len);
purities = zeros(noise_len, run_times, lambda_len, beta_len);
fmeasures = zeros(noise_len, run_times, lambda_len, beta_len);
ris = zeros(noise_len, run_times, lambda_len, beta_len); 
aris = zeros(noise_len, run_times, lambda_len, beta_len); 
time_costs = zeros(noise_len, run_times, lambda_len, beta_len);

for rat_idx = 1 : noise_len
    corrup_ratio = noise_ratios(rat_idx);    
    for time_idx = 1 : run_times
        X = XX;
        % Gussian noise
        rand('state', 10);
        gn = 0.3 * randn(row_size, col_size); 

        num = round(row_size * col_size * corrup_ratio);
        rand('state', 1);
        num_indexs = randperm(row_size * col_size);
        for idx =  1 : num
            row_index = fix(num_indexs(idx) / col_size);
            col_index = mod(num_indexs(idx), col_size);
            if(row_index == 0)
                rand('state', row_index);        
                row_index = randi(1, 1, row_size);                
            end
            if(col_index == 0)
                 rand('state', row_index);
                col_index = randi(1, 1, col_size);                 
            end          
            X(row_index, col_index) =  XX(row_index, col_index) + XX(row_index, col_index) .* gn(row_index, col_index);
        end
        
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
                    time_costs(rat_idx, time_idx, lmd_idx, beta_idx) = time_cost;

                    actual_ids = spectral_clustering(L, K);
                    acc = accuracy(gnd', actual_ids);
                    accs(rat_idx, time_idx, lmd_idx, beta_idx) = acc;

                    cluster_data = cell(1, K);
                    for pos_idx =  1 : K
                        cluster_data(1, pos_idx) = { gnd(actual_ids == pos_idx) };
                    end 
                    [nmi, purity, fmeasure, ri, ari] = calculate_results(class_labels, cluster_data);        
                     nmis(rat_idx, time_idx, lmd_idx, beta_idx) = nmi;
                    purities(rat_idx, time_idx, lmd_idx, beta_idx) = purity;
                    fmeasures(rat_idx, time_idx, lmd_idx, beta_idx) = fmeasure;
                    ris(rat_idx, time_idx, lmd_idx, beta_idx) = ri;
                    aris(rat_idx, time_idx, lmd_idx, beta_idx) = ari;
                    disp([lambda, beta, acc, nmi, ari, time_cost, iter]);
                    dlmwrite('alrr_coil_100_noise.txt', [rat_idx, time_idx, lambda, beta, acc, nmi, ari, time_cost, iter], '-append', 'delimiter', '\t', 'newline', 'pc');

                end
        end
    end
    for lmd_idx = 1 : length(lambdas)
        lambda = lambdas(lmd_idx);
        for beta_idx = 1 : length(betas)
             beta = betas(beta_idx);
            avg_acc = mean(accs(rat_idx, : , lmd_idx, beta_idx));
            std_acc = std(accs(rat_idx, : , lmd_idx, beta_idx));
            avg_nmi = mean(nmis(rat_idx, : , lmd_idx, beta_idx));
            std_nmi  = std(nmis(rat_idx, : , lmd_idx, beta_idx));
            avg_ari = mean(aris(rat_idx, : , lmd_idx, beta_idx));
            std_ari  = std(aris(rat_idx, : , lmd_idx, beta_idx));
            avg_time_cost = mean(time_costs(rat_idx, : , lmd_idx, beta_idx));
            std_time_cost  = std(time_costs(rat_idx, : , lmd_idx, beta_idx));
            dlmwrite('alrr_coil_100_avg_noise.txt', [rat_idx, lambda, beta, avg_acc, avg_nmi, avg_ari, std_acc, std_nmi, std_ari, avg_time_cost, std_time_cost], '-append', 'delimiter', '\t', 'newline', 'pc');
        end
    end
end




function [new_labels, num_current_clusters] = refresh_labels(current_lables, k_max)
%current_lables: 1 * N
% k_max: the maximum number of clusters
% Rename all indices of the data objects for the metrics 
% if the number of clusters is less than the maximum number of clusters
% For example: the index set of the current clusters is {1, 3, 5}, 
% and the updated index set of the current clusters is {1, 2, 3}
                    
    new_labels = current_lables;
    num_current_clusters = length(unique(current_lables));
    if num_current_clusters < k_max        
        label_result = 1;
        for label_idx = 1 : k_max            
            if(~isempty(find(current_lables == label_idx, 1)))
                if(label_result ~= label_idx)
                    new_labels(current_lables == label_idx) = label_result;
                end
                label_result = label_result + 1;
            end
        end     
    end


end


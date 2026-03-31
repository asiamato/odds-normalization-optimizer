function projected_vector = project_simplex(input_vector)
    % PROJECT_SIMPLEX - Projects an n-dimensional vector onto the probability simplex.
    % The feasible region is: { x : x is n-dim, x >= 0, sum(x) = 1 }
    %
    % Reference: Exact projection algorithm by Xiaojing Ye (2011).
    % Refactored for the Odds Normalization Optimizer pipeline.

    vec_length = length(input_vector); 
    is_found = false;
    
    % Sort the input vector in descending order
    sorted_vec = sort(input_vector, 'descend'); 
    cumulative_sum = 0;

    % Iterative threshold search
    for idx = 1:(vec_length - 1)
        cumulative_sum = cumulative_sum + sorted_vec(idx);
        tau = (cumulative_sum - 1) / idx;
        
        if tau >= sorted_vec(idx + 1)
            is_found = true;
            break;
        end
    end
        
    % Handle the boundary case
    if ~is_found
        tau = (cumulative_sum + sorted_vec(vec_length) - 1) / vec_length; 
    end

    % Apply the threshold tau and project onto the non-negative orthant
    projected_vector = max(input_vector - tau, 0);
end
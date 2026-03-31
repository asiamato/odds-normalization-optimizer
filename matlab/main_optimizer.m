% MAIN_OPTIMIZER - Simplex Projection and Information Theory Analysis
clear; clc; close all;

URL = 'https://www.football-data.co.uk/mmz4281/2425/I1.csv';
fprintf('Fetching data...\n');
data = readtable(webread(URL), 'VariableNamingRule', 'preserve');
data = rmmissing(data, 'DataVariables', {'B365H', 'FTR'});

% 1. Extract Quotes and Raw Probabilities
Q365 = [data.B365H, data.B365D, data.B365A];
P_raw = 1 ./ Q365;

% 2. Naive Normalization
P_norm = P_raw ./ sum(P_raw, 2);

% 3. Euclidean Simplex Projection
P_proj = zeros(size(P_raw));
for i = 1:size(P_raw, 1)
    P_proj(i, :) = project_simplex(P_raw(i, :)); % <-- Nome aggiornato qui!
end

% 4. KL Divergence (Generalized) from Original P_raw
% Formula: sum( P_model * log(P_model/P_raw) - P_model + P_raw )
KL_norm = sum(P_norm .* log(P_norm ./ P_raw) - P_norm + P_raw, 2);
KL_proj = sum(P_proj .* log(P_proj ./ P_raw) - P_proj + P_raw, 2);

fprintf('\n--- KL Divergence (Mean) ---\n');
fprintf('Naive Norm: %.6f\n', mean(KL_norm));
fprintf('Simplex Proj: %.6f\n', mean(KL_proj));

% 5. Quadratic Score (Brier)
O = zeros(size(P_raw));
O(strcmp(data.FTR, 'H'), 1) = 1;
O(strcmp(data.FTR, 'D'), 2) = 1;
O(strcmp(data.FTR, 'A'), 3) = 1;

BS_norm = mean(sum((P_norm - O).^2, 2));
BS_proj = mean(sum((P_proj - O).^2, 2));

fprintf('\n--- Brier Score (Mean) ---\n');
fprintf('Naive Norm: %.4f\n', BS_norm);
fprintf('Simplex Proj: %.4f\n', BS_proj);
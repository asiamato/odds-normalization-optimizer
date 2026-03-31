import pandas as pd
import numpy as np

class OptimizationEngine:
    """Handles advanced probability projections and information theory metrics."""
    
    @staticmethod
    def project_onto_simplex(y: np.ndarray) -> np.ndarray:
        """Projects an n-dim vector y onto the probability simplex."""
        m = len(y)
        s = np.sort(y)[::-1]
        tmpsum = 0.0
        bget = False
        
        for ii in range(m - 1):
            tmpsum += s[ii]
            tmax = (tmpsum - 1.0) / (ii + 1.0)
            if tmax >= s[ii + 1]:
                bget = True
                break
                
        if not bget:
            tmax = (tmpsum + s[-1] - 1.0) / m
            
        return np.maximum(y - tmax, 0.0)

    @staticmethod
    def kl_divergence_generalized(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Calculates Generalized KL Divergence between matrices."""
        # sum(P * log(P/Q) - P + Q)
        return np.sum(p * np.log(p / q) - p + q, axis=1)

class Pipeline:
    def run(self):
        url = 'https://www.football-data.co.uk/mmz4281/2425/I1.csv'
        print("Fetching dataset...")
        df = pd.read_csv(url).dropna(subset=['B365H', 'FTR'])
        
        quotes = df[['B365H', 'B365D', 'B365A']].values
        p_raw = 1.0 / quotes
        
        # 1. Naive Normalization
        p_norm = p_raw / np.sum(p_raw, axis=1, keepdims=True)
        
        # 2. Simplex Projection
        p_proj = np.zeros_like(p_raw)
        for i in range(len(p_raw)):
            p_proj[i] = OptimizationEngine.project_onto_simplex(p_raw[i])
            
        # 3. Information Theory (KL Divergence from original P)
        kl_norm = np.mean(OptimizationEngine.kl_divergence_generalized(p_norm, p_raw))
        kl_proj = np.mean(OptimizationEngine.kl_divergence_generalized(p_proj, p_raw))
        
        print("\n--- Information Theory (KL Divergence from P_raw) ---")
        print(f"Mean KL (Naive Normalization): {kl_norm:.6f}")
        print(f"Mean KL (Simplex Projection):  {kl_proj:.6f}")
        
        # 4. Evaluation (Brier Score)
        o_matrix = np.zeros_like(p_raw)
        o_matrix[df['FTR'] == 'H', 0] = 1
        o_matrix[df['FTR'] == 'D', 1] = 1
        o_matrix[df['FTR'] == 'A', 2] = 1
        
        bs_norm = np.mean(np.sum((p_norm - o_matrix)**2, axis=1))
        bs_proj = np.mean(np.sum((p_proj - o_matrix)**2, axis=1))
        
        print("\n--- Predictive Accuracy (Mean Brier Score) ---")
        print(f"BS (Naive Normalization): {bs_norm:.4f}")
        print(f"BS (Simplex Projection):  {bs_proj:.4f}")

if __name__ == "__main__":
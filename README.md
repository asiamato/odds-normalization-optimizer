# Odds Normalization Optimizer: Information-Theoretic Approach to Coherent Probabilities

## Abstract
In sports betting markets, implied probabilities derived directly from bookmakers' odds ($P_{raw} = 1/Q$) inherently sum to a value greater than 1. This "overround" violates the fundamental axioms of probability. This repository provides a polyglot software engineering pipeline (Python, MATLAB, R) to restore probability coherence. We move beyond naive normalization strategies by implementing **Euclidean Projection onto the Probability Simplex** and evaluating information retention using **Kullback-Leibler (KL) Divergence** and **Strictly Proper Scoring Rules**.

## Theoretical Framework

### 1. Restoring Coherence
A valid probability distribution must reside within the standard $(n-1)$-simplex $\Delta^{n-1}$, where $\sum_{i=1}^n p_i = 1$ and $p_i \ge 0$.

* **Naive Normalization:** The industry standard maps $P_{raw}$ to the simplex by scaling: 
    $$P_{norm} = \frac{P_{raw}}{\sum P_{raw}}$$
* **Simplex Projection (Optimization):** We formulate coherence restoration as a convex optimization problem. We seek the vector $P_{proj}$ in the simplex $\Delta$ that minimizes the Euclidean distance from the original $P_{raw}$:
    $$P_{proj} = \arg\min_{x \in \Delta} \frac{1}{2} ||x - P_{raw}||^2_2$$
    This is solved iteratively in our code using the exact projection algorithm.

### 2. Information-Theoretic Evaluation
To quantify how much original signal from the bookmaker is destroyed during the normalization processes, we employ the **Generalized Kullback-Leibler (KL) Divergence**:
$$D_{KL}(P || Q) = \sum \left( P \ln\left(\frac{P}{Q}\right) - P + Q \right)$$
*By comparing $D_{KL}(P_{norm} || P_{raw})$ and $D_{KL}(P_{proj} || P_{raw})$, we systematically evaluate which normalization method preserves the original predictive intent best.*

### 3. Predictive Accuracy
Finally, the models' actual predictive power against real-world full-time results is measured using the **Brier Score (Quadratic SPSR)**:
$$BS = \sum_{k=1}^K (p_k - o_k)^2$$

## Architecture & Multi-Language Implementation
This project is engineered to bridge advanced Operations Research with scalable Software Engineering:
- **Python (OOP):** Scalable optimization engine leveraging `numpy` vectorization.
- **MATLAB:** Matrix-optimized scientific scripts for rigorous empirical evaluation.
- **R:** Statistical pipeline using base data frames and functional operations.

## Application Note
This repository highlights a research-driven approach to Software Engineering, developed as a core portfolio project to demonstrate proficiency in convex optimization, information theory, and polyglot software design.
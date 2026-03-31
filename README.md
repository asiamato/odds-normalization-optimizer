# Odds Normalization Optimizer: Information-Theoretic and Convex Optimization Approaches to Coherent Probabilities

## Abstract
In sports betting markets, implied probabilities derived directly from bookmakers' odds inherently violate the axioms of probability by summing to a value strictly greater than 1. This project provides a polyglot software engineering pipeline (Python, MATLAB, R) designed to restore probability coherence. By moving beyond naive normalization strategies, we implement **Euclidean Projection onto the Probability Simplex** and rigorously evaluate information retention using **Kullback-Leibler (KL) Divergence**, **Shannon Entropy**, and **Strictly Proper Scoring Rules**.

---

## Theoretical Framework

### 1. Market Overround and Incoherence
Let $H_1, H_2, \dots, H_n$ be a set of mutually exclusive and exhaustive outcomes (e.g., Home, Draw, Away). A bookmaker offers a set of decimal odds $Q=\{q_1, q_2, \dots, q_n\}$. The raw implied probability for each outcome is defined as $p_i=\frac{1}{q_i}$.

To guarantee a risk-free profit (the *vigorish* or *margin*) and avoid exposing themselves to a Dutch Book, bookmakers design their odds such that the sum of implied probabilities exceeds 1. The market overround $\pi$ is formalized as:

$$\pi=\sum_{i=1}^n \frac{1}{q_i} - 1 > 0$$

The resulting vector $P_{raw}$ resides outside the standard $(n-1)$-simplex, rendering it an incoherent probability distribution.

### 2. Restoring Coherence: Convex Optimization
A coherent probability distribution must reside within the standard probability simplex $\Delta^{n-1}$, defined as:

$$\Delta^{n-1}=\{ x \in \mathbb{R}^n \mid \sum_{i=1}^n x_i=1, x_i \ge 0 \}$$

* **Naive Normalization (Industry Standard):** The standard approach simply scales the raw probabilities by their sum.
  
  $$P_{norm, i}=\frac{p_{raw, i}}{\sum_{j=1}^n p_{raw, j}}$$

* **Simplex Projection (Our Approach):** We formulate coherence restoration as a strictly convex optimization problem. We seek the vector $P_{proj} \in \Delta^{n-1}$ that minimizes the Euclidean distance (L2-norm) from the original $P_{raw}$:

  $$P_{proj}=\arg\min_{x \in \Delta^{n-1}} \frac{1}{2} ||x - P_{raw}||^2_2$$

  To solve this, we construct the Lagrangian incorporating the equality constraint and the non-negativity (Karush-Kuhn-Tucker - KKT) multipliers:

  $$\mathcal{L}(x, \lambda, \mu)=\frac{1}{2} ||x - P_{raw}||^2_2 - \lambda ( \sum_{i=1}^n x_i - 1 ) - \sum_{i=1}^n \mu_i x_i$$

  Our software implements an exact, iterative $\mathcal{O}(n \log n)$ projection algorithm (Ye, 2011) to resolve this Lagrangian, optimally stripping the bookmaker's margin while preserving the original continuous distribution shape.

### 3. Information-Theoretic Evaluation
To quantify the topological distance and information loss between the original betting market and our coherent models, we apply principles from Information Theory.

* **Shannon Entropy:** Measures the inherent uncertainty of the normalized predictive models.
  
  $$H(P)=- \sum_{i=1}^n P_i \ln(P_i)$$

* **Generalized Kullback-Leibler (KL) Divergence:** Since $P_{raw}$ does not sum to 1, standard relative entropy is undefined. We employ the Generalized KL Divergence to measure how much original signal is destroyed during normalization:

  $$D_{KL}(P || Q)=\sum_{i=1}^n ( P_i \ln(\frac{P_i}{Q_i}) - P_i + Q_i )$$

  *We systematically compare $D_{KL}(P_{norm} || P_{raw})$ and $D_{KL}(P_{proj} || P_{raw})$ to prove which algorithm best preserves the bookmaker's true predictive intent.*

### 4. Predictive Accuracy (Strictly Proper Scoring Rules)
Ultimately, the predictive power of the coherent distributions against real-world full-time outcomes ($O$, where $o_k=1$ if outcome $k$ occurs, $0$ otherwise) is evaluated using Strictly Proper Scoring Rules (SPSR).

* **Brier Score (Quadratic SPSR):** Measures the mean squared error of the probabilistic forecast.
  
  $$BS=\sum_{k=1}^n (P_k - o_k)^2$$

* **Logarithmic Score (Log SPSR):** Heavily penalizes confident but incorrect predictions based on the probability assigned to the realized event.
  
  $$LS=\ln(P_{winner})$$

---

## Architecture & Multi-Language Implementation
- **Python (OOP):** Scalable optimization engine leveraging `numpy` vectorization.
- **MATLAB:** Matrix-optimized scientific scripts for rigorous empirical evaluation.
- **R:** Statistical pipeline using base data frames and functional operations.

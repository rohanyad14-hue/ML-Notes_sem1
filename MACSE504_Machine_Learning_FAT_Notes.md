# MACSE504: Machine Learning - Master Comprehensive Notes for Final Assessment Test (FAT)
**Course Code:** MACSE504 | **Course Title:** Machine Learning | **L-T-P-C:** 3-0-2-4  
**Target:** Final Assessment Test (FAT), Continuous Assessments (CAT 1 & CAT 2), and Applied ML Engineering  
**Primary Textbooks & References:**  
1. L. Goel, *Artificial Intelligence: Concepts and Applications*, Wiley, 2021.  
2. Andreas Müller and Sarah Guido, *Introduction to Machine Learning with Python: A Guide for Data Scientists*, O'Reilly, 2016.  
3. T. Jo, *Machine Learning Foundations: Supervised, Unsupervised, and Advanced Learning*, Springer Nature, 2021.  
4. P. Dangeti, *Statistics for Machine Learning*, Packt Publishing, 2017.  
5. Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press.

---

## Detailed Table of Contents
1. [Module 1: Supervised Learning Algorithms](#module-1-supervised-learning-algorithms)
   - 1.1 Regression Fundamentals: Simple, Multiple & Polynomial Regression
   - 1.2 Mathematical Derivation of the OLS Closed-Form Normal Equation
   - 1.3 Gauss-Markov Assumptions & Multicollinearity Diagnostics (VIF)
   - 1.4 Regularization Techniques: Ridge ($L_2$) vs. Lasso ($L_1$) vs. ElasticNet
   - 1.5 Classification: Logistic Regression & Binary Cross-Entropy Loss
   - 1.6 K-Nearest Neighbors (KNN): Distance Metrics & Curse of Dimensionality
   - 1.7 Support Vector Machines (SVM): Maximum Margin Hyperplane & RBF Kernel
   - 1.8 Decision Trees: Splitting Criteria (Entropy, Information Gain, Gini Impurity)
   - 1.9 Random Forest: Bagging, Feature Subsampling & Out-of-Bag (OOB) Error
   - 1.10 Overfitting Diagnostics, Bias-Variance Tradeoff & Class Imbalance (SMOTE)
2. [Module 2: Unsupervised and Semi-Supervised Learning Algorithms](#module-2-unsupervised-and-semi-supervised-learning-algorithms)
   - 2.1 Clustering Foundations: K-Means Algorithm, K-Means++ & Silhouette Analysis
   - 2.2 Hierarchical Clustering: Agglomerative Linkages & Dendrogram Interpretation
   - 2.3 Association Rule Learning: Support, Confidence, Lift, Conviction
   - 2.4 The Apriori Algorithm & Anti-Monotonicity Principle
   - 2.5 FP-Growth Algorithm: FP-Tree Construction & Mining Conditional Pattern Bases
   - 2.6 Semi-Supervised Learning Framework: Smoothness & Cluster Assumptions
   - 2.7 Self-Training & Pseudo-Labeling Iterative Workflows
   - 2.8 Graph-Based Learning: Label Propagation (LPA) vs. Label Spreading
3. [Module 3: Reinforcement Learning (RL)](#module-3-reinforcement-learning-rl)
   - 3.1 Markov Decision Process (MDP): 5-Tuple Formulation & Cumulative Returns
   - 3.2 Value Functions: Bellman Expectation & Bellman Optimality Equations
   - 3.3 Value-Based RL: Q-Learning Update Rule, $\epsilon$-Greedy & Deep Q-Networks (DQN)
   - 3.4 Policy-Based RL: Policy Gradient Theorem & The REINFORCE Algorithm
   - 3.5 Advanced Policy Optimization: PPO (Clipped Surrogate Objective) & TRPO
   - 3.6 Model-Based RL vs. Model-Free RL: Dyna-Q Planning Architecture
4. [Module 4: Dimensionality Reduction](#module-4-dimensionality-reduction)
   - 4.1 Motivation: Curse of Dimensionality & Geometric Sparsity
   - 4.2 Principal Component Analysis (PCA): Covariance Matrix, Eigen-Decomposition & SVD
   - 4.3 Kernel PCA, Incremental PCA & Explained Variance Ratios
   - 4.4 Locally Linear Embedding (LLE): Local Neighborhood Weight Reconstruction
   - 4.5 t-Distributed Stochastic Neighbor Embedding (t-SNE): High vs. Low Dimension
   - 4.6 How the Student's t-Distribution (Cauchy) Resolves the Crowding Problem
5. [Module 5: ML Model Evaluation and Hyperparameter Tuning](#module-5-ml-model-evaluation-and-hyperparameter-tuning)
   - 5.1 Classification Metrics: Confusion Matrix, Precision, Recall, Specificity, F1-Score
   - 5.2 ROC Curves, Area Under Curve (AUC-ROC) & Precision-Recall Curves
   - 5.3 Regression Metrics: MAE, MSE, RMSE, $R^2$, and Adjusted $R^2$
   - 5.4 Cross-Validation Schemes: K-Fold, Stratified K-Fold, Group K-Fold, Time-Series Split
   - 5.5 Hyperparameter Tuning: Grid Search, Random Search & Bayesian Optimization
6. [Module 6: Contemporary Issues & High-Yield FAT Review](#module-6-contemporary-issues--high-yield-fat-review)
   - 6.1 Top 10 High-Frequency Theory Questions for FAT
   - 6.2 Master Machine Learning Algorithm Matrix Cheat Sheet

---

## Module 1: Supervised Learning Algorithms

### 1.1 Regression Fundamentals & Derivation of OLS Normal Equation
In multiple linear regression, the relationship between features $X \in \mathbb{R}^{n \times p}$ and target $y \in \mathbb{R}^{n \times 1}$ is:
$$y = X \beta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$
where $\beta = [\beta_0, \beta_1, \dots, \beta_p]^T$.

#### Formal Derivation of the OLS Closed-Form Estimator
The Ordinary Least Squares objective minimizes the Sum of Squared Residuals (SSR):
$$J(\beta) = \|y - X \beta\|^2 = (y - X \beta)^T (y - X \beta) = y^T y - 2 \beta^T X^T y + \beta^T X^T X \beta$$
To find the minimum, take the gradient with respect to $\beta$ and set it to zero:
$$\nabla_\beta J(\beta) = -2 X^T y + 2 X^T X \beta = 0$$
$$X^T X \beta = X^T y$$
Assuming $(X^T X)$ is non-singular (full column rank):
$$\hat{\beta} = (X^T X)^{-1} X^T y \quad \blacksquare$$

---

### 1.2 Regularization: Ridge ($L_2$) vs. Lasso ($L_1$) vs. ElasticNet
When features are highly collinear, $(X^T X)$ becomes near-singular, causing OLS coefficient variances to explode.
1. **Ridge Regression ($L_2$ Regularization):**
   $$J(\beta) = \sum_{i=1}^n (y_i - X_i \beta)^2 + \lambda \sum_{j=1}^p \beta_j^2 \implies \hat{\beta}_{\text{ridge}} = (X^T X + \lambda I)^{-1} X^T y$$
   - Adds $\lambda I$ to the diagonal, guaranteeing $(X^T X + \lambda I)$ is strictly invertible.
   - Shrinks coefficients continuously toward zero; never sets them exactly to zero.
2. **Lasso Regression ($L_1$ Regularization):**
   $$J(\beta) = \sum_{i=1}^n (y_i - X_i \beta)^2 + \lambda \sum_{j=1}^p |\beta_j|$$
   - The $L_1$ diamond constraint boundary produces sharp corners on the coordinate axes.
   - Drives uninformative coefficients **exactly to zero**, performing automatic **feature selection**.
3. **ElasticNet:** Combines both penalties: $\lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2$.

---

### 1.3 Classification: Logistic Regression & Loss Formulation
* **Sigmoid Activation Function:**
  $$P(Y = 1 \mid X) = \sigma(w^T X + b) = \frac{1}{1 + e^{-(w^T X + b)}}$$
* **Binary Cross-Entropy Loss (Log-Loss):**
  $$J(w) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$
  Derived directly via Maximum Likelihood Estimation on the Bernoulli distribution. Convex loss surface guarantees convergence to global minimum via gradient descent.

---

### 1.4 Support Vector Machines (SVM) & The Kernel Trick
Finds the optimal separating hyperplane maximizing the geometric margin $\frac{2}{\|w\|}$:
$$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i \quad \text{subject to } y_i(w^T x_i + b) \ge 1 - \xi_i, \; \xi_i \ge 0$$
* **Hyperparameter $C$:** Balances margin width and training error. High $C$ penalizes violations strictly (low bias, high variance); low $C$ allows softer margin.
* **Kernel Trick (Mercer's Theorem):** Implicitly maps data into high-dimensional Hilbert space where linear separation is possible, computing inner products via kernel functions without explicit coordinate mapping:
  - **Radial Basis Function (RBF / Gaussian):**
    $$K(x, z) = \exp\left( -\gamma \|x - z\|^2 \right)$$

---

### 1.5 Decision Trees & Random Forest
* **Splitting Criteria:**
  - **Entropy:** $H(S) = -\sum p_i \log_2(p_i)$
  - **Information Gain (ID3):** $IG(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$
  - **Gini Impurity (CART):** $Gini(S) = 1 - \sum p_i^2$
* **Random Forest (Ensemble Learning):**
  - **Bagging (Bootstrap Aggregation):** Trains $B$ independent trees on random resamples with replacement.
  - **Feature Subsampling:** Considers only a random subset of $\sqrt{p}$ features at each split to decorrelate individual trees. Reduces variance without increasing bias.

---

## Module 2: Unsupervised and Semi-Supervised Learning Algorithms

### 2.1 K-Means Clustering & Silhouette Analysis
* **Objective Function (WCSS / Inertia):**
  $$J = \sum_{j=1}^k \sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$
* **K-Means++ Initialization:** Picks first centroid at random; picks subsequent centroids with probability proportional to squared Euclidean distance to closest existing center: $P(x) = \frac{D(x)^2}{\sum D(x')^2}$. Prevents poor local minima.
* **Silhouette Coefficient:**
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, +1]$$
  where $a(i)$ is mean intra-cluster distance, and $b(i)$ is mean distance to nearest neighboring cluster. $s(i) \approx 1$ implies optimal clustering.

---

### 2.2 Association Rule Learning: Apriori vs. FP-Growth
* **Formulas:**
  - $\text{Support}(A \to B) = P(A \cup B) = \frac{\text{Transactions containing } A \text{ and } B}{\text{Total Transactions}}$
  - $\text{Confidence}(A \to B) = P(B \mid A) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$
  - $\text{Lift}(A \to B) = \frac{\text{Confidence}(A \to B)}{\text{Support}(B)} = \frac{P(A \cap B)}{P(A)P(B)}$
* **Apriori Algorithm:** Relies on the **Anti-Monotonicity Principle**: If an itemset is infrequent, all of its supersets are guaranteed to be infrequent. Generates $k$-candidates and prunes via multiple full database scans.
* **FP-Growth Algorithm:** Builds a compact **FP-Tree** (Frequent Pattern Tree) in memory using only 2 full database scans. Mines conditional pattern bases recursively without expensive candidate generation.

---

### 2.3 Semi-Supervised Learning: LPA vs. Label Spreading
* **Label Propagation Algorithm (LPA):**
  - Builds graph affinity matrix $W_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$.
  - Row-normalizes to transition matrix $T = D^{-1} W$.
  - Iteratively updates: $Y^{(t+1)} = T Y^{(t)}$. True labeled nodes are hard-clamped at every step.
* **Label Spreading:**
  - Uses normalized symmetric Laplacian: $S = D^{-1/2} W D^{-1/2}$.
  - Introduces regularization parameter $\alpha \in (0, 1)$:
    $$Y^{(t+1)} = \alpha S Y^{(t)} + (1 - \alpha) Y^{(0)}$$
  - Softens label constraints, providing high robustness against mislabeled noise.

---

## Module 3: Reinforcement Learning (RL)

### 3.1 Markov Decision Process (MDP) & Bellman Equations
Formulated as $(S, A, P, R, \gamma)$ with discount factor $\gamma \in [0, 1)$:
* **Bellman Expectation Equation for $V^\pi(s)$:**
  $$V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} P(s', r \mid s, a) \left[ r + \gamma V^\pi(s') \right]$$
* **Bellman Optimality Equation for $Q^*(s, a)$:**
  $$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')$$

---

### 3.2 Q-Learning vs. Policy Gradients
* **Q-Learning (Model-Free, Value-Based, Off-Policy TD Control):**
  $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$
  - Action selection: $\epsilon$-greedy.
  - **Deep Q-Networks (DQN):** Stabilized by **Experience Replay** (breaking sequential correlations) and a frozen **Target Network**.
* **Policy Gradient (Model-Free, Policy-Based):**
  $$\nabla_\theta J(\theta) = E_{\pi_\theta}\left[ \nabla_\theta \ln \pi_\theta(a \mid s) Q^{\pi_\theta}(s, a) \right]$$
* **Proximal Policy Optimization (PPO):** Clips surrogate objective to prevent destructively large updates:
  $$L^{CLIP}(\theta) = \hat{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
  where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$.

---

## Module 4: Dimensionality Reduction

### 4.1 Principal Component Analysis (PCA)
Projects data onto orthogonal principal axes of maximum variance:
1. Standardize features: $X \leftarrow (X - \mu)/\sigma$.
2. Covariance Matrix: $\Sigma = \frac{1}{n} X^T X \in \mathbb{R}^{p \times p}$.
3. Solve Eigen-system: $\Sigma v_i = \lambda_i v_i$.
4. Sort eigenvalues $\lambda_1 \ge \lambda_2 \dots \ge \lambda_p$. Explained variance ratio of $i$-th component is $\frac{\lambda_i}{\sum \lambda_j}$.
5. Project data: $Z = X W_k$.

---

### 4.2 t-Distributed Stochastic Neighbor Embedding (t-SNE)
* Converts high-dimensional Euclidean distances into probabilities using Gaussian distribution: $p_{j|i}$.
* In low-dimensional 2D/3D space, models similarity using the heavy-tailed **Student's t-distribution with 1 df (Cauchy distribution)**:
  $$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$
* **Solving the Crowding Problem:** In high dimensions, space expands exponentially. The Cauchy distribution's heavy tails allow moderate-distance points in high dimensions to be mapped significantly farther apart in 2D without inflating KL-divergence penalties.
* Objective: Minimizes Kullback-Leibler (KL) divergence via gradient descent:
  $$KL(P \parallel Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

---

## Module 5: ML Model Evaluation and Hyperparameter Tuning

### 5.1 Classification Metrics
* **Confusion Matrix:** True Positives ($TP$), False Positives ($FP$), True Negatives ($TN$), False Negatives ($FN$).
* **Precision:** $\frac{TP}{TP + FP}$ (Minimizes false alarms, e.g., Spam filter).
* **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ (Minimizes missed positives, e.g., Tumor detection).
* **Specificity:** $\frac{TN}{TN + FP}$
* **F1-Score:** Harmonic mean: $F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
* **ROC-AUC:** True Positive Rate vs. False Positive Rate ($1 - \text{Specificity}$) across all discrimination thresholds.

### 5.2 Cross-Validation & Hyperparameter Tuning
* **Stratified K-Fold:** Guarantees each fold preserves identical minority-class target proportions.
* **Group K-Fold:** Ensures observations from identical entities (patients, institutions) never appear simultaneously in both train and validation splits (prevents data leakage).
* **Random Search vs. Grid Search:** Bergstra and Bengio demonstrated that Random Search finds equal or superior models in a fraction of the compute time because most ML algorithms have low effective dimensionality (only a subset of hyperparameters dictate performance).

---

## Module 6: Contemporary Issues & High-Yield FAT Review

### 6.1 Top 10 High-Frequency FAT Exam Questions
1. **OLS Normal Equation Derivation:** Derive $\hat{\beta} = (X^T X)^{-1} X^T y$ and explain the Gauss-Markov assumptions.
2. **Ridge vs. Lasso Regularization:** Draw the geometry of $L_1$ and $L_2$ constraints and explain why Lasso enforces sparsity.
3. **SVM Formulation:** State the primal optimization problem with slack variables and explain the role of the RBF kernel.
4. **Apriori vs. FP-Growth:** Detail how the FP-Tree eliminates candidate itemset generation.
5. **Bellman Optimality Equations:** Formulate $V^*(s)$ and $Q^*(s, a)$ and derive the Q-Learning TD update.
6. **t-SNE vs. PCA:** Compare linear vs. non-linear dimensionality reduction and explain how the Cauchy distribution resolves the crowding problem.
7. **Cross-Validation Selection:** When must you use Stratified K-Fold vs. Group K-Fold vs. Time-Series Split?
8. **Handling Imbalanced Datasets:** Explain SMOTE algorithm mechanics and contrast with random undersampling.
9. **K-Means Convergence & Silhouette Analysis:** Define the Silhouette coefficient formula and explain its interpretation.
10. **Policy Gradient Theorem:** State the theorem and explain the intuition behind the REINFORCE algorithm.

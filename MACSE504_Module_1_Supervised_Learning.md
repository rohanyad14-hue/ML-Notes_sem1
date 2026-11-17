# MACSE504: Machine Learning - Module 1
# Comprehensive Study Notes: Supervised Learning Algorithms

<div class="meta-box">
<strong>Course:</strong> MACSE504 Machine Learning (ETH 1.0) | School of Computer Science and Engineering<br>
<strong>Prescribed Textbooks:</strong><br>
1. L. Goel, <em>Artificial Intelligence: Concepts and Applications</em>, Wiley, 2021.<br>
2. A. Müller & S. Guido, <em>Introduction to Machine Learning with Python</em>, O'Reilly, 2016.<br>
3. T. Jo, <em>Machine Learning Foundations</em>, Springer Nature, 2021.<br>
4. P. Dangeti, <em>Statistics for Machine Learning</em>, Packt Publishing, 2017.<br>
<strong>VIT Exam Target:</strong> Complete Module 1 Master Manual for Classification, Regression, Regularization, Overfitting, and Imbalanced Data Handling.
</div>

---

## 📺 Curated Video Lectures by Indian Professors (Under 20 Minutes)
* **Krish Naik - Complete Logistic Regression Intuition & Math (18:12 min):** [Watch on YouTube](https://www.youtube.com/watch?v=L_x_KK_925A)
* **CampusX - Support Vector Machine (SVM) Geometric Intuition (19:00 min):** [Watch on YouTube](https://www.youtube.com/watch?v=efR1C6CvhmE)
* **Gate Smashers - Decision Tree Classification with Numerical Example (14:15 min):** [Watch on YouTube](https://www.youtube.com/watch?v=coOTEc-0OGw)
* **CampusX - Random Forest & Ensemble Learning (18:40 min):** [Watch on YouTube](https://www.youtube.com/watch?v=ok2s1vV9XW0)
* **Krish Naik - OLS Normal Equations & Multiple Linear Regression (18:24 min):** [Watch on YouTube](https://www.youtube.com/watch?v=zITIFTsivN8)
* **CampusX - Bias-Variance Decomposition & SMOTE (18:15 min):** [Watch on YouTube](https://www.youtube.com/watch?v=Bqgm4MQvEVM)

---

## 1. Classification Algorithms Overview

### 1.1 Logistic Regression
Models posterior probability $P(Y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$.
- **Loss Function:** Binary Cross-Entropy Loss $\mathcal{L}(\mathbf{w}) = -\frac{1}{n} \sum [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$.
- **Gradient Update:** $\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{n} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$.

### 1.2 K-Nearest Neighbors (KNN)
Non-parametric, instance-based memory learner.
- **Distance Metrics:** Euclidean $\|\mathbf{u} - \mathbf{v}\|_2$, Manhattan $\|\mathbf{u} - \mathbf{v}\|_1$, Minkowski $\left(\sum |u_i - v_i|^p\right)^{1/p}$.
- **Decision Rule:** $y = \arg\max_c \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbb{I}(y_i = c)$.
- **Time Complexity:** $\mathcal{O}(n \cdot d)$ per test query without indexing (KD-Trees optimize to $\mathcal{O}(\log n)$).

### 1.3 Support Vector Machines (SVM)
Finds maximum margin hyperplane $w^T x + b = 0$ separating classes with margin $M = \frac{2}{\|w\|}$.
- **Soft Margin Optimization:** $\min \frac{1}{2}\|w\|^2 + C \sum \xi_i$ subject to $y_i(w^Tx_i + b) \ge 1 - \xi_i$.
- **Kernel Trick:** Computes inner products in implicit Hilbert spaces: RBF Kernel $K(x, z) = \exp(-\gamma \|x - z\|^2)$.

### 1.4 Decision Trees & Random Forests
Recursive greedy binary splitting of feature space.
- **Impurity Criteria:** Entropy $H(S) = -\sum p_i \log_2 p_i$, Gini Impurity $G(S) = 1 - \sum p_i^2$.
- **Information Gain:** $\text{IG}(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$.
- **Random Forest:** Bagging (Bootstrap Aggregation) + Random Feature Subsampling ($m = \sqrt{d}$) to decorrelate individual trees and reduce model variance.

---

## 2. Regression Algorithms Overview

### 2.1 Ordinary Least Squares (OLS)
Closed-form normal equation:
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
Valid under Gauss-Markov assumptions (Linearity, Exogeneity $E[\epsilon|X]=0$, Spherical errors $\text{Var}(\epsilon|X)=\sigma^2 I$, Non-collinear $\mathbf{X}$).

### 2.2 Regularization: Ridge vs Lasso vs Elastic Net
- **Ridge ($L_2$):** $\min \text{RSS} + \lambda \|\boldsymbol{\beta}\|_2^2 \implies \hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^Ty$ (Shrinks parameters smoothly).
- **Lasso ($L_1$):** $\min \text{RSS} + \lambda \|\boldsymbol{\beta}\|_1 \implies$ Enforces exact sparsity due to sharp corners on diamond constraint boundary.
- **Elastic Net:** Combines $L_1$ and $L_2$ penalties with mixing parameter $\alpha \in [0, 1]$.

---

## 3. High-Yield Solved Problem Summary (VIT Exam Focus)

### Problem: Bias-Variance Error Decomposition
Total expected test MSE error decomposes as:
$$E[(y - \hat{f}(x))^2] = \text{Bias}^2(\hat{f}(x)) + \text{Var}(\hat{f}(x)) + \sigma^2$$
- Underfitting $\implies$ High Bias, Low Variance.
- Overfitting $\implies$ Low Bias, High Variance.
- SMOTE generates synthetic minority samples: $\mathbf{x}_{\text{new}} = \mathbf{x}_i + \delta \cdot (\mathbf{x}_{zi} - \mathbf{x}_i)$ where $\delta \sim \mathcal{U}(0, 1)$.


# MACSE504 - Machine Learning: Module 1
## Topic 07: Regularization Techniques (Ridge, Lasso & Elastic Net)

<div class="meta-box">
<strong>Course:</strong> MACSE504 - Machine Learning (ETH 1.0) | School of Computer Science & Engineering<br>
<strong>Prescribed Textbooks:</strong> Müller & Guido, <em>Introduction to Machine Learning with Python</em> (O'Reilly 2016); Dangeti, <em>Statistics for Machine Learning</em> (Packt 2017).<br>
<strong>VIT Exam Target:</strong> 10-Mark Detailed Answer for L1 vs L2 Shrinkage, Geometric Sparsity Proof, Closed-Form Ridge Derivation, and Sub-gradient Lasso Optimization.
</div>

---

### 1. The Regularization Principle & Motivation

Ordinary Least Squares (OLS) can suffer from high variance when features are collinear ($\det(\mathbf{X}^T\mathbf{X}) \approx 0$) or when $d > n$ (overparameterized regime). Under these conditions, weights $\beta_j$ explode in magnitude, leading to catastrophic overfitting.

Regularization introduces a penalty parameter $\lambda > 0$ to constrain parameter complexity, trading a small amount of bias for a substantial reduction in model variance.

#### 1.1 Ridge Regression ($L_2$ Regularization)
Ridge regression introduces an $L_2$-norm penalty on the weight vector $\boldsymbol{\beta}$:
$$\mathcal{L}_{\text{Ridge}}(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \frac{\lambda}{2} \|\boldsymbol{\beta}\|_2^2 = \frac{1}{2n} (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + \frac{\lambda}{2} \boldsymbol{\beta}^T\boldsymbol{\beta}$$

<div class="formula-box">
<strong>Derivation of Closed-Form Ridge Solution:</strong><br>
Taking the gradient with respect to $\boldsymbol{\beta}$:
$$\nabla_{\boldsymbol{\beta}} \mathcal{L}_{\text{Ridge}} = -\frac{1}{n}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + \lambda \boldsymbol{\beta} = \mathbf{0}$$
$$\frac{1}{n} \mathbf{X}^T\mathbf{X}\boldsymbol{\beta} + \lambda \boldsymbol{\beta} = \frac{1}{n}\mathbf{X}^T\mathbf{y}$$
$$\left(\mathbf{X}^T\mathbf{X} + n\lambda \mathbf{I}\right) \boldsymbol{\beta} = \mathbf{X}^T\mathbf{y} \quad \implies \quad \hat{\boldsymbol{\beta}}_{\text{Ridge}} = \left(\mathbf{X}^T\mathbf{X} + \lambda' \mathbf{I}\right)^{-1} \mathbf{X}^T\mathbf{y}$$
<em>Key Mathematical Property:</em> Even if $\mathbf{X}^T\mathbf{X}$ is singular (non-invertible), adding $\lambda' \mathbf{I}$ (where $\lambda' > 0$) guarantees that all eigenvalues of $(\mathbf{X}^T\mathbf{X} + \lambda' \mathbf{I})$ are strictly positive ($\mu_i + \lambda' > 0$). Thus, the inverse **always exists**.
</div>

#### 1.2 Lasso Regression ($L_1$ Regularization)
Lasso (**L**east **A**bsolute **S**hrinkage and **S**election **O**perator) applies an $L_1$-norm penalty:
$$\mathcal{L}_{\text{Lasso}}(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_1 = \frac{1}{2n} \sum_{i=1}^n (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda \sum_{j=1}^d |\beta_j|$$

Because the absolute value function $|x|$ has a sharp corner at zero, it is non-differentiable at $\beta_j = 0$. Therefore, Lasso has **no closed-form matrix inverse** and requires sub-gradient descent or Coordinate Descent.

#### 1.3 Elastic Net
Elastic Net combines both $L_1$ and $L_2$ penalties with mixing parameter $\alpha \in [0, 1]$:
$$\mathcal{L}_{\text{ElasticNet}}(\boldsymbol{\beta}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \left( \alpha \|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2} \|\boldsymbol{\beta}\|_2^2 \right)$$
It overcomes Lasso's limitation when features are highly correlated (where Lasso arbitrarily selects one and zeroes out the rest) by encouraging a **grouping effect** while maintaining sparsity.

---

### 2. Geometric Interpretation: Why Lasso Produces Sparse Solutions

<div class="exam-box">
<strong>Classic 10-Mark Question:</strong> <em>"Explain with neat geometric diagrams why $L_1$ regularization acts as a feature selector by setting coefficients exactly to zero, while $L_2$ regularization only shrinks them towards zero."</em>
</div>

Constrained optimization formulations (by Lagrange duality):
$$\min_{\boldsymbol{\beta}} \text{RSS}(\boldsymbol{\beta}) \quad \text{subject to} \quad \begin{cases} \|\boldsymbol{\beta}\|_2^2 \le t & \text{(Ridge)} \\ \|\boldsymbol{\beta}\|_1 \le t & \text{(Lasso)} \end{cases}$$

1. **Lasso Constraint Region ($L_1$):** In 2 dimensions ($\beta_1, \beta_2$), the constraint $|\beta_1| + |\beta_2| \le t$ forms a **rhombus (diamond)** with sharp corners on the coordinate axes: $(t, 0), (-t, 0), (0, t), (0, -t)$.
2. **Ridge Constraint Region ($L_2$):** The constraint $\beta_1^2 + \beta_2^2 \le t$ forms a **smooth circle (hypersphere)** centered at the origin.
3. **RSS Contours:** The elliptical contours represent level sets of the quadratic OLS cost function $\text{RSS}(\boldsymbol{\beta})$, expanding outward from the unconstrained OLS solution $\hat{\boldsymbol{\beta}}_{\text{OLS}}$.
4. **Point of Tangency:** The optimal regularized solution is the first point where the expanding RSS elliptical contour touches the constraint boundary:
   - For Lasso, the expanding ellipse is mathematically far more likely to strike one of the **sharp corners** of the diamond located precisely on an axis ($\beta_1 = 0$ or $\beta_2 = 0$). This sets coefficients **strictly to zero**, performing automatic **feature selection**.
   - For Ridge, because the spherical boundary is smooth and differentiable everywhere with no sharp corners, tangency typically occurs at a point where neither $\beta_1$ nor $\beta_2$ is zero. It shrinks weights asymptotically but never eliminates them.

---

### 3. Step-by-Step Solved Numerical Problem (Coordinate Descent for Lasso)

<div class="exam-box">
<strong>VIT Exam Problem [10 Marks]:</strong><br>
Consider an orthonormal single-feature dataset where $X^T X = 1$ and $X^T y = 4.0$.<br>
1. Compute the unconstrained OLS solution $\hat{\beta}_{\text{OLS}}$.<br>
2. Derive and compute the Ridge estimate for $\lambda = 2.0$.<br>
3. Apply the Soft-Thresholding operator to determine the Lasso estimate for $\lambda = 1.0, 3.0,$ and $5.0$.
</div>

#### Step 1: OLS Estimate
$$\hat{\beta}_{\text{OLS}} = (X^T X)^{-1} X^T y = (1)^{-1}(4.0) = \mathbf{4.0}$$

#### Step 2: Ridge Solution
$$\hat{\beta}_{\text{Ridge}} = (X^T X + \lambda)^{-1} X^T y = \frac{X^T y}{1 + \lambda} = \frac{4.0}{1 + 2.0} = \frac{4.0}{3.0} \approx \mathbf{1.333}$$
Notice: The coefficient shrunk from 4.0 down to 1.333, but remains strictly non-zero.

#### Step 3: Lasso Solution via Soft-Thresholding Operator
For orthonormal features, the Lasso objective is:
$$\min_{\beta} \frac{1}{2}(4.0 - \beta)^2 + \lambda |\beta|$$
The sub-gradient optimality condition yields the **Soft-Thresholding Operator** $S_{\lambda}(\hat{\beta}_{\text{OLS}})$:
$$\hat{\beta}_{\text{Lasso}} = \text{sign}(\hat{\beta}_{\text{OLS}}) \cdot \max(0, |\hat{\beta}_{\text{OLS}}| - \lambda) = \begin{cases} \hat{\beta}_{\text{OLS}} - \lambda & \text{if } \hat{\beta}_{\text{OLS}} > \lambda \\ 0 & \text{if } |\hat{\beta}_{\text{OLS}}| \le \lambda \\ \hat{\beta}_{\text{OLS}} + \lambda & \text{if } \hat{\beta}_{\text{OLS}} < -\lambda \end{cases}$$

Evaluating for given $\lambda$ values:
1. **Case A ($\lambda = 1.0$):**
   $$|\hat{\beta}_{\text{OLS}}| = 4.0 > 1.0 \implies \hat{\beta}_{\text{Lasso}} = 4.0 - 1.0 = \mathbf{3.0}$$
2. **Case B ($\lambda = 3.0$):**
   $$|\hat{\beta}_{\text{OLS}}| = 4.0 > 3.0 \implies \hat{\beta}_{\text{Lasso}} = 4.0 - 3.0 = \mathbf{1.0}$$
3. **Case C ($\lambda = 5.0$):**
   $$|\hat{\beta}_{\text{OLS}}| = 4.0 \le 5.0 \implies \hat{\beta}_{\text{Lasso}} = \mathbf{0.0}$$
*Conclusion:* When regularization penalty $\lambda \ge |\hat{\beta}_{\text{OLS}}|$, Lasso forcefully clamps the parameter to zero, eliminating that feature.

---

### 4. Comparison Table: Ridge vs Lasso vs Elastic Net

| Dimension | Ridge Regression ($L_2$) | Lasso Regression ($L_1$) | Elastic Net ($L_1 + L_2$) |
|:---|:---|:---|:---|
| **Penalty Term** | $\lambda \sum \beta_j^2$ | $\lambda \sum \|\beta_j\|$ | $\lambda_1 \sum \|\beta_j\| + \lambda_2 \sum \beta_j^2$ |
| **Solution Form** | Closed-form analytic solution | Numerical / Coordinate Descent | Numerical / Coordinate Descent |
| **Feature Sparsity** | No (retains all features) | **Yes** (sets coefficients to 0) | **Yes** (selective sparsity) |
| **Multicollinearity Handling** | Stabilizes matrix inversion | Arbitrarily picks one feature | Groups correlated features together |
| **High Dimensions ($d > n$)** | Estimates all $d$ features | Selects at most $n$ features | Can select $> n$ features |
| **Computational Cost** | $\mathcal{O}(d^3)$ matrix inversion | Iterative coordinate updates | Iterative coordinate updates |

---

### 5. Curated Video Lectures ($\le$ 20 Mins)

<div class="video-box">
<strong>High-Yield Video References by Renowned Indian Educators:</strong><br>
1. <strong>CampusX:</strong> <em>"Ridge Regression Math Intuition & Cost Function"</em> (Duration: 18:50 min) - Intuitive closed-form derivation and geometric shrinking.<br>
<a href="https://www.youtube.com/watch?v=0k5i5vQ_V7s">https://www.youtube.com/watch?v=0k5i5vQ_V7s</a><br><br>
2. <strong>Krish Naik:</strong> <em>"Lasso Regression and Feature Selection Mathematical Proof"</em> (Duration: 16:32 min) - Visual contour demonstration of why Lasso sets weights to zero.<br>
<a href="https://www.youtube.com/watch?v=9lhyPdmgR0I">https://www.youtube.com/watch?v=9lhyPdmgR0I</a><br><br>
3. <strong>Gate Smashers:</strong> <em>"Regularization in Machine Learning (Ridge & Lasso)"</em> (Duration: 11:20 min) - Clear summary of bias-variance reduction through penalty terms.<br>
<a href="https://www.youtube.com/watch?v=bQhA4Q0dIUk">https://www.youtube.com/watch?v=bQhA4Q0dIUk</a>
</div>

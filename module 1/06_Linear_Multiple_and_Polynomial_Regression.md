
# MACSE504 - Machine Learning: Module 1
## Topic 06: Linear, Multiple, and Polynomial Regression

<div class="meta-box">
<strong>Course:</strong> MACSE504 - Machine Learning (ETH 1.0) | School of Computer Science & Engineering<br>
<strong>Prescribed Textbooks:</strong> Dangeti, <em>Statistics for Machine Learning</em> (Packt 2017); Müller & Guido, <em>Introduction to Machine Learning with Python</em> (O'Reilly 2016); L. Goel, <em>Artificial Intelligence</em> (Wiley 2021).<br>
<strong>VIT Exam Target:</strong> 10-Mark Detailed Answer for Ordinary Least Squares (OLS) Derivation, Gauss-Markov Assumptions, Multiple Linear Regression, and Polynomial Bias-Variance Overfitting.
</div>

---

### 1. Theoretical Foundations & Model Formulations

Regression analysis models the conditional expectation $E[Y|X]$ of a continuous dependent target $y \in \mathbb{R}$ given an explanatory vector of features $\mathbf{x} = [x_1, x_2, \dots, x_d]^T \in \mathbb{R}^d$.

#### 1.1 Simple Linear Regression (SLR)
For a single independent variable $x$, the regression model is:
$$y_i = \beta_0 + \beta_1 x_i + \epsilon_i, \quad i = 1, 2, \dots, n$$
where $\beta_0$ is the intercept, $\beta_1$ is the slope parameter, and $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ is the random error term.

#### 1.2 Multiple Linear Regression (MLR)
In matrix notation, for $n$ observations and $d$ features (augmenting $x_{i0} = 1$ for intercept):
$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$
where:
$$\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}_{n \times 1}, \quad \mathbf{X} = \begin{bmatrix} 1 & x_{11} & \cdots & x_{1d} \\ 1 & x_{21} & \cdots & x_{2d} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & \cdots & x_{nd} \end{bmatrix}_{n \times (d+1)}, \quad \boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_d \end{bmatrix}_{(d+1) \times 1}$$

<div class="formula-box">
<strong>Derivation of the OLS Normal Equation:</strong><br>
The Ordinary Least Squares (OLS) criterion minimizes the Residual Sum of Squares (RSS):
$$S(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$
Taking the gradient with respect to $\boldsymbol{\beta}$ and equating to zero vector $\mathbf{0}$:
$$\nabla_{\boldsymbol{\beta}} S(\boldsymbol{\beta}) = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{0}$$
$$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y} \quad \implies \quad \hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
Provided $\mathbf{X}^T\mathbf{X}$ has full column rank $(d+1)$ (no exact multicollinearity).
</div>

#### 1.3 Polynomial Regression
When relationships between independent features and target are non-linear, polynomial basis expansions map feature $x$ to higher-degree powers:
$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_p x^p + \epsilon$$
Although non-linear in feature $x$, the model remains **linear in its parameters** $\beta_j$, allowing solution via the identical OLS normal equation with polynomial Vandermonde design matrix:
$$\mathbf{X}_{\text{poly}} = \begin{bmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^p \\ 1 & x_2 & x_2^2 & \cdots & x_2^p \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_n & x_n^2 & \cdots & x_n^p \end{bmatrix}$$

---

### 2. Gauss-Markov Theorem & Statistical Assumptions

The Gauss-Markov theorem states that under 5 fundamental assumptions, the OLS estimator $\hat{\boldsymbol{\beta}}$ is **BLUE** (**B**est **L**inear **U**nbiased **E**stimator):

1. **Linearity in Parameters:** The relationship between dependent and independent variables is linear in the parameters $\boldsymbol{\beta}$.
2. **Strict Exogeneity:** $E[\boldsymbol{\epsilon} | \mathbf{X}] = \mathbf{0}$. The error term has conditional expectation equal to zero given feature matrix $\mathbf{X}$.
3. **No Multicollinearity:** $\text{Rank}(\mathbf{X}) = d+1 < n$. Columns of $\mathbf{X}$ are linearly independent; $(\mathbf{X}^T\mathbf{X})^{-1}$ exists.
4. **Spherical Errors (Homoscedasticity & No Autocorrelation):**
   $$\text{Var}(\boldsymbol{\epsilon}|\mathbf{X}) = \sigma^2 \mathbf{I}_n \implies \begin{cases} \text{Var}(\epsilon_i) = \sigma^2 & \forall i \text{ (Homoscedasticity)} \\ \text{Cov}(\epsilon_i, \epsilon_j) = 0 & \forall i \ne j \text{ (No serial correlation)} \end{cases}$$
5. **Normality (for hypothesis testing):** $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_n)$. Required for valid $t$-tests and $F$-tests.

#### Multicollinearity Diagnostic: Variance Inflation Factor (VIF)
When features are highly correlated, standard errors of $\hat{\beta}_j$ explode. VIF for feature $j$ is calculated as:
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
where $R_j^2$ is the coefficient of determination when regressing $x_j$ on all other remaining features.
- $\text{VIF} = 1$: No correlation.
- $1 < \text{VIF} < 5$: Moderate, acceptable correlation.
- $\text{VIF} > 5$ (or $> 10$): Severe multicollinearity; parameter estimates become wildly unstable and must be remediated via Ridge/Lasso regularization or PCA.

---

### 3. Step-by-Step Solved 10-Mark Numerical Problem (VIT Format)

<div class="exam-box">
<strong>VIT Fall Semester Exam Question [10 Marks]:</strong><br>
Consider a dataset consisting of $n=5$ real-estate transactions with feature $x_1$ (Area in hundreds of sq.ft) and target $y$ (Price in lakhs INR):<br>
$$\mathcal{D} = \{(1, 2), (2, 3), (3, 5), (4, 4), (5, 6)\}$$
1. Formulate the design matrix $\mathbf{X}$ and target vector $\mathbf{y}$. (2 Marks)
2. Compute $\mathbf{X}^T\mathbf{X}$, $(\mathbf{X}^T\mathbf{X})^{-1}$, and $\mathbf{X}^T\mathbf{y}$. (3 Marks)
3. Derive the OLS parameter vector $\hat{\boldsymbol{\beta}} = [\hat{\beta}_0, \hat{\beta}_1]^T$. (2 Marks)
4. Compute the Coefficient of Determination $R^2$ and interpret the goodness-of-fit. (3 Marks)
</div>

#### Step 1: Matrix Formulation
$$\mathbf{X} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4 \\ 1 & 5 \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} 2 \\ 3 \\ 5 \\ 4 \\ 6 \end{bmatrix}$$

#### Step 2: Compute $\mathbf{X}^T\mathbf{X}$ and $\mathbf{X}^T\mathbf{y}$
$$\mathbf{X}^T\mathbf{X} = \begin{bmatrix} 1 & 1 & 1 & 1 & 1 \\ 1 & 2 & 3 & 4 & 5 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4 \\ 1 & 5 \end{bmatrix} = \begin{bmatrix} n & \sum x_i \\ \sum x_i & \sum x_i^2 \end{bmatrix}$$
- $n = 5$
- $\sum x_i = 1 + 2 + 3 + 4 + 5 = 15$
- $\sum x_i^2 = 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 1 + 4 + 9 + 16 + 25 = 55$
$$\mathbf{X}^T\mathbf{X} = \begin{bmatrix} 5 & 15 \\ 15 & 55 \end{bmatrix}$$

Determinant: $\det(\mathbf{X}^T\mathbf{X}) = (5)(55) - (15)(15) = 275 - 225 = 50 \ne 0$ (Invertible).

Inverting the $2 \times 2$ matrix:
$$(\mathbf{X}^T\mathbf{X})^{-1} = \frac{1}{50} \begin{bmatrix} 55 & -15 \\ -15 & 5 \end{bmatrix} = \begin{bmatrix} 1.1 & -0.3 \\ -0.3 & 0.1 \end{bmatrix}$$

Now compute $\mathbf{X}^T\mathbf{y}$:
$$\mathbf{X}^T\mathbf{y} = \begin{bmatrix} \sum y_i \\ \sum x_i y_i \end{bmatrix} = \begin{bmatrix} 2 + 3 + 5 + 4 + 6 \\ 1(2) + 2(3) + 3(5) + 4(4) + 5(6) \end{bmatrix} = \begin{bmatrix} 20 \\ 2 + 6 + 15 + 16 + 30 \end{bmatrix} = \begin{bmatrix} 20 \\ 69 \end{bmatrix}$$

#### Step 3: Solve for $\hat{\boldsymbol{\beta}}$
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T\mathbf{y} = \begin{bmatrix} 1.1 & -0.3 \\ -0.3 & 0.1 \end{bmatrix} \begin{bmatrix} 20 \\ 69 \end{bmatrix} = \begin{bmatrix} 1.1(20) - 0.3(69) \\ -0.3(20) + 0.1(69) \end{bmatrix} = \begin{bmatrix} 22 - 20.7 \\ -6 + 6.9 \end{bmatrix} = \begin{bmatrix} 1.3 \\ 0.9 \end{bmatrix}$$
Hence:
$$\hat{\beta}_0 = 1.3, \quad \hat{\beta}_1 = 0.9$$
Regression Equation: $\hat{y} = 1.3 + 0.9x$.

#### Step 4: Goodness-of-Fit Evaluation ($R^2$)
Calculate fitted values $\hat{y}_i$, residuals $e_i = y_i - \hat{y}_i$, and total variance $\bar{y} = \frac{20}{5} = 4.0$:

| $i$ | $x_i$ | $y_i$ | $\hat{y}_i = 1.3 + 0.9x_i$ | $e_i = y_i - \hat{y}_i$ | $e_i^2$ | $(y_i - \bar{y})^2$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | 2 | 2.2 | -0.2 | 0.04 | $(2-4)^2 = 4$ |
| 2 | 2 | 3 | 3.1 | -0.1 | 0.01 | $(3-4)^2 = 1$ |
| 3 | 3 | 5 | 4.0 | +1.0 | 1.00 | $(5-4)^2 = 1$ |
| 4 | 4 | 4 | 4.9 | -0.9 | 0.81 | $(4-4)^2 = 0$ |
| 5 | 5 | 6 | 5.8 | +0.2 | 0.04 | $(6-4)^2 = 4$ |
| **Sum** | **15** | **20** | **20.0** | **0.0** | $\text{SS}_{\text{res}} = \mathbf{1.90}$ | $\text{SS}_{\text{tot}} = \mathbf{10.0}$ |

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{1.90}{10.0} = 1 - 0.19 = \mathbf{0.81} \text{ (81.0\%)}$$
**Interpretation:** 81% of the total variance in house prices is explained by floor area using this linear regression model.

---

### 4. Algorithmic Pseudocode & Implementation

```python
import numpy as np

class MultipleLinearRegressionOLS:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        # Augment design matrix with ones for intercept
        n_samples = X.shape[0]
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((n_samples, 1)), X])
        else:
            X_aug = X
            
        # Closed-form solution: beta = (X^T * X)^(-1) * X^T * y
        XtX = np.dot(X_aug.T, X_aug)
        Xty = np.dot(X_aug.T, y)
        beta = np.linalg.solve(XtX, Xty) # Numerically stable over inv()
        
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        return self
        
    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
```

---

### 5. Curated Video Lectures ($\le$ 20 Mins)

<div class="video-box">
<strong>High-Yield Video References by Renowned Indian Educators:</strong><br>
1. <strong>Krish Naik:</strong> <em>"Multiple Linear Regression with Mathematical Intuition"</em> (Duration: 18:24 min) - Covers OLS derivation, normal equations, and assumptions.<br>
<a href="https://www.youtube.com/watch?v=zITIFTsivN8">https://www.youtube.com/watch?v=zITIFTsivN8</a><br><br>
2. <strong>CampusX:</strong> <em>"Polynomial Regression from Scratch | Overfitting & Underfitting"</em> (Duration: 19:45 min) - Clear geometric intuition on higher degree polynomials.<br>
<a href="https://www.youtube.com/watch?v=4b78o4zJ0lQ">https://www.youtube.com/watch?v=4b78o4zJ0lQ</a><br><br>
3. <strong>Gate Smashers:</strong> <em>"Linear Regression in Machine Learning with Numerical Example"</em> (Duration: 12:40 min) - Quick calculation walkthrough perfect for university semester exams.<br>
<a href="https://www.youtube.com/watch?v=owI7zxCqNY0">https://www.youtube.com/watch?v=owI7zxCqNY0</a>
</div>

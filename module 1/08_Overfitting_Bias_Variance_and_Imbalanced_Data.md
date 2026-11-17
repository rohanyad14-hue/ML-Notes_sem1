
# MACSE504 - Machine Learning: Module 1
## Topic 08: Overfitting, Bias-Variance Tradeoff & Imbalanced Data Handling

<div class="meta-box">
<strong>Course:</strong> MACSE504 - Machine Learning (ETH 1.0) | School of Computer Science & Engineering<br>
<strong>Prescribed Textbooks:</strong> Müller & Guido, <em>Introduction to Machine Learning with Python</em> (O'Reilly 2016); Dangeti, <em>Statistics for Machine Learning</em> (Packt 2017); Jo, <em>Machine Learning Foundations</em> (Springer 2021).<br>
<strong>VIT Exam Target:</strong> 10-Mark Detailed Answer for Bias-Variance Decomposition Proof, Overfitting vs Underfitting Diagnostics, SMOTE Synthetic Sampling Algorithm, and Cost-Sensitive Weighting.
</div>

---

### 1. Mathematical Bias-Variance Decomposition

<img src="images/ml_bias_variance.svg" alt="Bias Variance Tradeoff Curve">

Let true data-generating process be $y = f(\mathbf{x}) + \epsilon$, where $E[\epsilon] = 0$ and $\text{Var}(\epsilon) = \sigma^2$ (irreducible noise). Given a training dataset $\mathcal{D}$, let $\hat{f}(\mathbf{x}; \mathcal{D})$ be the model's estimate.

The expected Mean Squared Error (MSE) at an unseen test point $\mathbf{x}$ across all realizations of training sets $\mathcal{D}$ decomposes into three orthogonal components:
$$E_{\mathcal{D}}\left[ (y - \hat{f}(\mathbf{x}))^2 \right] = \text{Bias}^2(\hat{f}(\mathbf{x})) + \text{Var}(\hat{f}(\mathbf{x})) + \sigma^2$$

<div class="formula-box">
<strong>Full Mathematical Derivation:</strong><br>
Let expected prediction over all training sets be $\bar{f}(\mathbf{x}) = E_{\mathcal{D}}[\hat{f}(\mathbf{x})]$.
$$\begin{aligned}
E[(y - \hat{f}(\mathbf{x}))^2] &= E\left[ ( (f(\mathbf{x}) - \hat{f}(\mathbf{x})) + \epsilon )^2 \right] \\
&= E[(f(\mathbf{x}) - \hat{f}(\mathbf{x}))^2] + 2 E[(f(\mathbf{x}) - \hat{f}(\mathbf{x})) \epsilon] + E[\epsilon^2]
\end{aligned}$$
Since $\epsilon$ is independent of training data $\mathcal{D}$, $E[(f(\mathbf{x}) - \hat{f}(\mathbf{x}))\epsilon] = 0$, and $E[\epsilon^2] = \sigma^2$:
$$E[(y - \hat{f}(\mathbf{x}))^2] = E\left[ ( (f(\mathbf{x}) - \bar{f}(\mathbf{x})) + (\bar{f}(\mathbf{x}) - \hat{f}(\mathbf{x})) )^2 \right] + \sigma^2$$
Expanding the square:
$$= (f(\mathbf{x}) - \bar{f}(\mathbf{x}))^2 + E[(\bar{f}(\mathbf{x}) - \hat{f}(\mathbf{x}))^2] + 2(f(\mathbf{x}) - \bar{f}(\mathbf{x})) \underbrace{E[\bar{f}(\mathbf{x}) - \hat{f}(\mathbf{x})]}_{= 0} + \sigma^2$$
$$= \underbrace{(f(\mathbf{x}) - E[\hat{f}(\mathbf{x})])^2}_{\mathbf{Bias}^2} + \underbrace{E[(\hat{f}(\mathbf{x}) - E[\hat{f}(\mathbf{x})])^2]}_{\mathbf{Variance}} + \underbrace{\sigma^2}_{\mathbf{Irreducible\ Noise}}$$
</div>

#### Clinical Model Diagnostics: Underfitting vs Overfitting
- **High Bias (Underfitting):** Model makes rigid, overly simplistic assumptions (e.g., linear model on quadratic data).
  - Diagnostic: High Training Error $\approx$ High Validation Error.
  - Remedy: Increase model capacity (higher polynomial degree, deeper trees, add more features).
- **High Variance (Overfitting):** Model captures random noise and idiosyncratic training sample fluctuations.
  - Diagnostic: Low Training Error $\lll$ High Validation Error (large generalization gap).
  - Remedy: Regularization ($L_1/L_2$), pruning, ensemble averaging (bagging), collect more data, dropout.

---

### 2. Handling Imbalanced Datasets

In high-stakes domains (fraud detection, cancer diagnosis, fault detection), target classes are heavily skewed (e.g., 99% Negative, 1% Positive). Standard accuracy is misleading ($\text{Accuracy} = 99\%$ by predicting all negatives).

#### 2.1 Resampling Strategies
1. **Random Under-Sampling (RUS):** Randomly discard majority class samples until balance is reached.
   - *Limitation:* Discards potentially vital informational samples.
2. **Random Over-Sampling (ROS):** Duplicate minority class samples.
   - *Limitation:* Causes severe overfitting by replicating exact points.
3. **Synthetic Minority Over-sampling Technique (SMOTE):** Synthesizes artificial samples in feature space along the line segments connecting minority instances to their $k$-nearest neighbors.

<div class="formula-box">
<strong>SMOTE Algorithm Mathematical Formulation:</strong><br>
For each minority class sample $\mathbf{x}_i$:
1. Find its $k$ nearest neighbors within the minority class using Euclidean distance.
2. Randomly select one neighbor $\mathbf{x}_{zi}$ from the $k$ neighbors.
3. Generate a synthetic sample $\mathbf{x}_{\text{new}}$ along the vector difference:
$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \delta \cdot (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \text{where } \delta \sim \mathcal{U}(0, 1)$$
</div>

#### 2.2 Cost-Sensitive Learning & Class Weighting
Instead of altering dataset distributions, modify the loss function by assigning penalty weight $w_c$ inversely proportional to class frequency $N_c$:
$$w_c = \frac{N}{C \cdot N_c}$$
For binary cross-entropy with positive weight $w_1 > 1$:
$$\mathcal{L}_{\text{weighted}} = -\sum_{i=1}^n \left[ w_1 y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

---

### 3. Step-by-Step Solved 10-Mark Numerical Problem (VIT Format)

<div class="exam-box">
<strong>VIT Exam Problem [10 Marks]:</strong><br>
A medical screening diagnostic system for rare cardiac arrhythmia is evaluated on 1,000 patients.<br>
Actual Distribution: 50 Arrhythmia positive ($y=1$), 950 Healthy negative ($y=0$).<br>
Classifier Prediction Matrix:<br>
- True Positives ($\text{TP}$) = 40<br>
- False Positives ($\text{FP}$) = 60<br>
- False Negatives ($\text{FN}$) = 10<br>
- True Negatives ($\text{TN}$) = 890<br><br>
1. Compute standard Accuracy and explain why it is deceptive. (2 Marks)<br>
2. Compute Precision, Recall (Sensitivity), and $F_1$-score. (3 Marks)<br>
3. Calculate the balanced class weights $w_0$ and $w_1$ for cost-sensitive training. (2 Marks)<br>
4. Synthesize one new SMOTE sample given minority point $\mathbf{x}_1 = [2.0, 5.0]$, neighbor $\mathbf{x}_2 = [4.0, 8.0]$, and random draw $\delta = 0.4$. (3 Marks)
</div>

#### Step 1: Accuracy & Fallacy
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}} = \frac{40 + 890}{1000} = \frac{930}{1000} = \mathbf{93.0\%}$$
*Fallacy:* Even a naive dummy classifier predicting all 1000 patients as healthy would achieve $95.0\%$ accuracy while missing **100% of the sick patients**.

#### Step 2: Evaluation Metrics
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{40}{40 + 60} = \frac{40}{100} = \mathbf{0.40} \text{ (40.0\%)}$$
$$\text{Recall (Sensitivity)} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{40}{40 + 10} = \frac{40}{50} = \mathbf{0.80} \text{ (80.0\%)}$$
$$F_1\text{-score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \cdot \frac{0.40 \cdot 0.80}{0.40 + 0.80} = \frac{0.64}{1.20} \approx \mathbf{0.533} \text{ (53.3\%)}$$

#### Step 3: Balanced Class Weights
Total samples $N = 1000$, classes $C = 2$, $N_0 = 950$, $N_1 = 50$:
$$w_0 = \frac{N}{2 \cdot N_0} = \frac{1000}{2 \cdot 950} = \frac{1000}{1900} \approx \mathbf{0.526}$$
$$w_1 = \frac{N}{2 \cdot N_1} = \frac{1000}{2 \cdot 50} = \frac{1000}{100} = \mathbf{10.0}$$
*Impact:* Misclassifying a positive arrhythmia case incurs **19 times greater penalty** than misclassifying a healthy case ($10.0 / 0.526 \approx 19$).

#### Step 4: Synthetic SMOTE Sample Generation
Given:
$$\mathbf{x}_1 = \begin{bmatrix} 2.0 \\ 5.0 \end{bmatrix}, \quad \mathbf{x}_2 = \begin{bmatrix} 4.0 \\ 8.0 \end{bmatrix}, \quad \delta = 0.4$$
Difference vector:
$$\mathbf{d} = \mathbf{x}_2 - \mathbf{x}_1 = \begin{bmatrix} 4.0 - 2.0 \\ 8.0 - 5.0 \end{bmatrix} = \begin{bmatrix} 2.0 \\ 3.0 \end{bmatrix}$$
Synthetic sample calculation:
$$\mathbf{x}_{\text{new}} = \mathbf{x}_1 + \delta \cdot \mathbf{d} = \begin{bmatrix} 2.0 \\ 5.0 \end{bmatrix} + 0.4 \begin{bmatrix} 2.0 \\ 3.0 \end{bmatrix} = \begin{bmatrix} 2.0 + 0.8 \\ 5.0 + 1.2 \end{bmatrix} = \mathbf{\begin{bmatrix} 2.8 \\ 6.2 \end{bmatrix}}$$
The synthetic sample $[2.8, 6.2]$ lies directly on the segment joining the two minority samples in feature space.

---

### 4. SMOTE Algorithm Pseudocode

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote_oversample(X_minority, N_synthetic_needed, k_neighbors=5):
    n_samples, n_features = X_minority.shape
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X_minority)
    _, indices = nbrs.kneighbors(X_minority)
    
    synthetic_samples = []
    for _ in range(N_synthetic_needed):
        # Pick random minority sample
        i = np.random.randint(0, n_samples)
        # Pick random neighbor among the k-nearest (exclude index 0 which is self)
        nn_idx = indices[i, np.random.randint(1, k_neighbors + 1)]
        
        # Interpolate with uniform random delta in [0, 1]
        delta = np.random.rand()
        x_new = X_minority[i] + delta * (X_minority[nn_idx] - X_minority[i])
        synthetic_samples.append(x_new)
        
    return np.array(synthetic_samples)
```

---

### 5. Curated Video Lectures ($\le$ 20 Mins)

<div class="video-box">
<strong>High-Yield Video References by Renowned Indian Educators:</strong><br>
1. <strong>Krish Naik:</strong> <em>"Handling Imbalanced Dataset with SMOTE and Class Weights"</em> (Duration: 17:40 min) - Practical walkthrough of resampling vs cost-sensitive techniques.<br>
<a href="https://www.youtube.com/watch?v=YMPMZmlH5Bo">https://www.youtube.com/watch?v=YMPMZmlH5Bo</a><br><br>
2. <strong>CampusX:</strong> <em>"Bias Variance Tradeoff in Machine Learning Explained"</em> (Duration: 18:15 min) - Deep mathematical intuition behind irreducible error and model capacity.<br>
<a href="https://www.youtube.com/watch?v=Bqgm4MQvEVM">https://www.youtube.com/watch?v=Bqgm4MQvEVM</a><br><br>
3. <strong>Gate Smashers:</strong> <em>"Overfitting and Underfitting with Bias Variance Tradeoff"</em> (Duration: 11:45 min) - Exam-focused summary of diagnostics and learning curves.<br>
<a href="https://www.youtube.com/watch?v=euJ34zX0m9I">https://www.youtube.com/watch?v=euJ34zX0m9I</a>
</div>

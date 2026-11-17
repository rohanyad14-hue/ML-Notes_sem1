# MACSE504: Machine Learning - Module 5
# Comprehensive Study Notes: Model Evaluation and Hyperparameter Tuning

<div class="meta-box">
<strong>Course:</strong> MACSE504 Machine Learning (ETH 1.0) | School of Computer Science and Engineering<br>
<strong>Prescribed Textbooks:</strong><br>
1. A. Müller & S. Guido, <em>Introduction to Machine Learning with Python</em>, O'Reilly, 2016.<br>
2. P. Dangeti, <em>Statistics for Machine Learning</em>, Packt Publishing, 2017.<br>
3. T. Jo, <em>Machine Learning Foundations</em>, Springer Nature, 2021.<br>
<strong>VIT Exam Target:</strong> 10-Mark Detailed Answers for Confusion Matrix Metrics, ROC-AUC Analysis, Cross-Validation Types, and Bayesian Optimization vs Grid/Random Search.
</div>

---

## 📺 Curated Video Lectures by Indian Professors (Under 20 Minutes)
* **Krish Naik - Performance Metrics in Machine Learning (Confusion Matrix) (17:40 min):** [Watch on YouTube](https://www.youtube.com/watch?v=WPpnws_tKzE)  
  *Core Focus:* Precision, Recall, Specificity, F1-Score, and type I vs type II trade-offs.
* **CampusX (Nitish Singh) - ROC-AUC Curve Explained with Math & Intuition (18:20 min):** [Watch on YouTube](https://www.youtube.com/watch?v=F_S2iLw5j80)  
  *Core Focus:* TPR vs FPR thresholds, Area Under Curve interpretation as ranking probability.
* **Gate Smashers (Varun Singla) - Cross Validation & K-Fold with Examples (12:15 min):** [Watch on YouTube](https://www.youtube.com/watch?v=fSytzGwwBVw)  
  *Core Focus:* Train-test leakage, Stratified K-Fold, and model generalization variance.

---

## 1. Classification Metrics & Interpretation

<div class="diagram-box">
  <img src="images/ml_confusion_matrix.svg" style="max-width:560px;" alt="Confusion Matrix and ROC Curve">
  <div class="diagram-caption">Figure 5.1: Binary Confusion Matrix quadrant and Receiver Operating Characteristic (ROC) space.</div>
</div>

### 1.1 Complete Formula Suite
* **Accuracy:** Overall correctness:
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
* **Precision (Positive Predictive Value):** Exactness of positive predictions:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
* **Recall (Sensitivity / True Positive Rate):** Completeness of positive detection:
  $$\text{Recall} = \text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
* **Specificity (True Negative Rate):** Correctness of negative identification:
  $$\text{Specificity} = \text{TNR} = \frac{\text{TN}}{\text{TN} + \text{FP}} \implies \text{FPR} = 1 - \text{Specificity} = \frac{\text{FP}}{\text{TN} + \text{FP}}$$
* **$F_1$-Score:** Harmonic mean balancing Precision and Recall:
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

### 1.2 Multi-Class Averaging Strategies
For $C$ classes:
1. **Macro Average:** Unweighted arithmetic mean across classes. Treats all classes equally regardless of frequency:
   $$\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^C F_{1, c}$$
2. **Micro Average:** Aggregates global TP, FP, FN across all classes before computing metric:
   $$\text{Micro } F_1 = \frac{2 \sum \text{TP}_c}{2 \sum \text{TP}_c + \sum \text{FP}_c + \sum \text{FN}_c}$$
3. **Weighted Average:** Weights class $F_1$ by its support $N_c$:
   $$\text{Weighted } F_1 = \frac{1}{N} \sum_{c=1}^C N_c F_{1, c}$$

---

## 2. Regression Evaluation Metrics

* **Mean Absolute Error (MAE):** Linear penalty, robust to outliers:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
* **Mean Squared Error (MSE):** Quadratic penalty, heavily penalizes large errors:
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
* **Root Mean Squared Error (RMSE):** In identical units as the target:
  $$\text{RMSE} = \sqrt{\text{MSE}}$$
* **$R^2$ (Coefficient of Determination):**
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$
* **Adjusted $R^2$:** Penalizes superfluous features:
  $$R_{\text{adj}}^2 = 1 - \left[\frac{(1 - R^2)(n - 1)}{n - p - 1}\right]$$
  where $n$ is sample size and $p$ is number of features.

---

## 3. Cross-Validation Schemes

| CV Scheme | Mechanism | Ideal Use-Case |
|:---|:---|:---|
| **Standard K-Fold** | Shuffles and splits data into $K$ equal folds. | Balanced datasets with I.I.D. samples. |
| **Stratified K-Fold** | Preserves class target proportion in each fold. | **Imbalanced classification** (mandatory). |
| **Group K-Fold** | Ensures samples from the same group/entity do not leak across train and test. | Multi-patient medical records, multi-device logs. |
| **Time-Series Split** | Forward-chaining walk-forward validation ($t_1 \to t_2 \to t_3$). | Sequential, temporal, or financial time-series data. |

---

## 4. Hyperparameter Tuning Strategies

1. **Grid Search (`GridSearchCV`):** Exhaustive search over specified parameter grid. $\mathcal{O}(\prod_i |S_i|)$ complexity. Exponentially expensive.
2. **Randomized Search (`RandomizedSearchCV`):** Samples fixed number of configurations $N$ from continuous distributions. Highly efficient at finding optimal regions with significantly fewer iterations.
3. **Bayesian Optimization:** Constructs a probabilistic surrogate model (Gaussian Process) of objective function $f(\theta)$ and uses an **Acquisition Function** (Expected Improvement, Upper Confidence Bound) to balance exploration of uncertain regions with exploitation of known optima.

---

## 5. Step-by-Step Solved 10-Mark Numerical Problem (VIT Format)

<div class="exam-box">
<strong>VIT Exam Problem [10 Marks]:</strong><br>
A 3-class classifier (Classes: A, B, C) produces the following Confusion Matrix on 100 test samples:<br>
$$\begin{array}{c|ccc}
\text{Actual \ Predicted} & \text{Class A} & \text{Class B} & \text{Class C} \\
\hline
\text{Class A} & 30 & 5 & 5 \\
\text{Class B} & 3 & 25 & 2 \\
\text{Class C} & 2 & 3 & 25
\end{array}$$<br>
(a) Compute the overall Classification Accuracy. (2 Marks)<br>
(b) Calculate Precision, Recall, and $F_1$-score for Class A. (3 Marks)<br>
(c) Calculate Precision, Recall, and $F_1$-score for Class B and Class C. (3 Marks)<br>
(d) Calculate the Macro-averaged $F_1$ score of the classifier. (2 Marks)
</div>

#### Step 1: Overall Classification Accuracy
Diagonal entries are correct predictions:
$$\text{Correct} = 30 + 25 + 25 = 80$$
$$\text{Total} = 100$$
$$\text{Accuracy} = \frac{80}{100} = \mathbf{0.80} \text{ (80.0\%)}$$

#### Step 2: Metrics for Class A
- $\text{TP}_A = 30$
- Row sum for A (Actual A) $= 30 + 5 + 5 = 40 \implies \text{FN}_A = 5 + 5 = 10$
- Column sum for A (Predicted A) $= 30 + 3 + 2 = 35 \implies \text{FP}_A = 3 + 2 = 5$

$$\text{Precision}_A = \frac{\text{TP}_A}{\text{TP}_A + \text{FP}_A} = \frac{30}{35} \approx \mathbf{0.8571} \text{ (85.71\%)}$$
$$\text{Recall}_A = \frac{\text{TP}_A}{\text{TP}_A + \text{FN}_A} = \frac{30}{40} = \mathbf{0.7500} \text{ (75.00\%)}$$
$$F_{1, A} = 2 \cdot \frac{0.8571 \times 0.7500}{0.8571 + 0.7500} = \frac{1.28565}{1.6071} \approx \mathbf{0.8000} \text{ (80.00\%)}$$

#### Step 3: Metrics for Class B & Class C
**For Class B:**
- $\text{TP}_B = 25$
- Actual B (Row sum) $= 3 + 25 + 2 = 30 \implies \text{FN}_B = 3 + 2 = 5$
- Predicted B (Column sum) $= 5 + 25 + 3 = 33 \implies \text{FP}_B = 5 + 3 = 8$
$$\text{Precision}_B = \frac{25}{33} \approx \mathbf{0.7576}, \quad \text{Recall}_B = \frac{25}{30} \approx \mathbf{0.8333}$$
$$F_{1, B} = 2 \cdot \frac{0.7576 \times 0.8333}{0.7576 + 0.8333} = \frac{1.2626}{1.5909} \approx \mathbf{0.7936}$$

**For Class C:**
- $\text{TP}_C = 25$
- Actual C (Row sum) $= 2 + 3 + 25 = 30 \implies \text{FN}_C = 2 + 3 = 5$
- Predicted C (Column sum) $= 5 + 2 + 25 = 32 \implies \text{FP}_C = 5 + 2 = 7$
$$\text{Precision}_C = \frac{25}{32} \approx \mathbf{0.7812}, \quad \text{Recall}_C = \frac{25}{30} \approx \mathbf{0.8333}$$
$$F_{1, C} = 2 \cdot \frac{0.7812 \times 0.8333}{0.7812 + 0.8333} = \frac{1.3020}{1.6145} \approx \mathbf{0.8064}$$

#### Step 4: Macro-Averaged $F_1$ Score
$$\text{Macro } F_1 = \frac{F_{1, A} + F_{1, B} + F_{1, C}}{3} = \frac{0.8000 + 0.7936 + 0.8064}{3} = \frac{2.4000}{3} = \mathbf{0.8000} \text{ (80.0\%)}$$

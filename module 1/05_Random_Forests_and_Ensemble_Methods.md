# MACSE504: Machine Learning - Module 1 | Topic 5
# Random Forest: Bagging, Feature Subsampling & Out-of-Bag (OOB) Estimation
**Course:** MACSE504 Machine Learning | **Credits:** 4 | **Target:** 10-Mark VIT Examination Questions  
**Prescribed Textbooks:**  
1. Andreas Müller & Sarah Guido, *Introduction to Machine Learning with Python*, O'Reilly, 2016 (Chapter 2).  
2. L. Goel, *Artificial Intelligence: Concepts and Applications*, Wiley, 2021.  
3. P. Dangeti, *Statistics for Machine Learning*, Packt Publishing, 2017.

---

## 📺 Curated Video Lectures by Indian Professors (Under 20 Minutes)
* **CampusX (Nitish Singh) - Random Forest Algorithm & Bagging Explained (19 min):** [Watch on YouTube](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)  
  *Core Focus:* Ensemble learning, bootstrap sampling, and tree decorrelation.
* **Krish Naik - Out-of-Bag (OOB) Score & Feature Importance (15 min):** [Watch on YouTube](https://www.youtube.com/watch?v=sQ870aK8mSU)  
  *Core Focus:* The $e^{-1} \approx 36.8\%$ OOB mathematical proof and Gini importance.
* **Gate Smashers (Varun Singla) - Bagging vs. Boosting Comparison (12 min):** [Watch on YouTube](https://www.youtube.com/watch?v=tfg_y_z_G5Y)  
  *Core Focus:* Parallel vs. sequential ensembles, variance reduction vs. bias reduction.

---

## 1. Random Forest Mechanics (Leo Breiman, 2001)

A **Random Forest** is an ensemble of $B$ unpruned decision trees trained via **Bootstrap Aggregating (Bagging)** and **Random Feature Subsampling**.

### 1.1 The Two Pillars of Random Forest Randomization
1. **Bootstrap Aggregation (Row Sampling with Replacement):** Each tree $b$ is trained on a bootstrap sample $D_b$ of size $N$ drawn uniformly with replacement from training dataset $D$.
2. **Random Feature Subsampling (Column Sampling):** At each node split, the algorithm selects a random subset of $m$ features out of total $p$ features:
   - For Classification: $m = \lfloor \sqrt{p} \rfloor$
   - For Regression: $m = \lfloor p / 3 \rfloor$

### 1.2 Mathematical Variance Reduction
If we average $B$ identically distributed (but correlated) random variables, each with variance $\sigma^2$ and pairwise correlation $\rho$:
$$\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
* As $B \to \infty$, the second term vanishes: $\lim_{B \to \infty} \text{Var} = \rho \sigma^2$.
* Random feature subsampling reduces the pairwise tree correlation $\rho \downarrow$, driving down the overall ensemble variance!

---

## 2. Mathematical Proof of Out-of-Bag (OOB) Fraction

<div class="box-proof">
<h4>Mathematical Proof: ~36.8% of Samples are Out-of-Bag (OOB)</h4>
Consider a dataset of $N$ instances. In bootstrap sampling, each draw picks an instance with probability $\frac{1}{N}$.
The probability that a specific instance $x_i$ is <strong>not picked</strong> in a single draw is:
$$P(\text{not picked in 1 draw}) = 1 - \frac{1}{N}$$
For a bootstrap sample of size $N$ drawn independently with replacement:
$$P(\text{not picked in } N \text{ draws}) = \left( 1 - \frac{1}{N} \right)^N$$
Taking the limit as $N \to \infty$:
$$\lim_{N \to \infty} \left( 1 - \frac{1}{N} \right)^N = \frac{1}{e} \approx 0.367879 \approx \mathbf{36.8\%}$$
<strong>Physical Meaning:</strong>
For every individual tree, approximately $36.8\%$ of the training instances are never seen. These "Out-of-Bag" instances act as an inherent, built-in validation test set for evaluating generalization error without requiring explicit cross-validation splits! $\blacksquare$
</div>

---

## 3. Model 10-Mark Solved Problems for VIT Examination

<div class="box-example">
<h4>10-Mark Solved Problem 5.1: Random Forest Feature Importance & Ensemble Voting</h4>
<strong>Question:</strong> Explain the Mean Decrease in Impurity (MDI) mechanism for computing Feature Importance in Random Forests. In a credit card fraud detection ensemble of 5 trees, output probabilities for positive class (Fraud) are: $[0.85, 0.70, 0.45, 0.90, 0.65]$.
<br>(a) Compute the ensemble prediction using Soft Voting (Average Probability).
<br>(b) Compute the ensemble prediction using Hard Voting (Threshold 0.50).
<br>(c) Why is Soft Voting preferred over Hard Voting in practice? (10 Marks - VIT Lab 2 Pattern)
<br><br>
<strong>(a) Soft Voting (Average Probability):</strong>
$$P_{\text{ensemble}} = \frac{0.85 + 0.70 + 0.45 + 0.90 + 0.65}{5} = \frac{3.55}{5} = \mathbf{0.710 \text{ (71.0\%)}}$$
Since $P = 0.710 > 0.50$, the predicted class is **Fraud (Positive)**.
<br><br>
<strong>(b) Hard Voting (Majority Rule):</strong>
- Tree 1: $0.85 \ge 0.50 \to \text{Fraud}$
- Tree 2: $0.70 \ge 0.50 \to \text{Fraud}$
- Tree 3: $0.45 < 0.50 \to \text{Non-Fraud}$
- Tree 4: $0.90 \ge 0.50 \to \text{Fraud}$
- Tree 5: $0.65 \ge 0.50 \to \text{Fraud}$
<br>
Votes for Fraud $= 4$, Votes for Non-Fraud $= 1$.
$$\text{Prediction} = \mathbf{\text{Fraud (4 vs 1 Majority)}}$$
<br>
<strong>(c) Soft Voting Superiority:</strong>
Soft voting assigns higher weight to trees that are highly confident ($0.90$) and attenuates marginally confident predictions ($0.51$). It accounts for predictive uncertainty, consistently yielding superior ROC-AUC and calibration.
</div>
